# FastAPI REST API wrapper for PII/PCI detection using BERT
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import re
import json
import hashlib
from datetime import datetime
import uvicorn
import torch
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification, pipeline
)
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for request/response
class ChatMessage(BaseModel):
    text: str = Field(..., description="Chat message text to analyze for PII/PCI", min_length=1, max_length=5000)
    user_id: Optional[str] = Field(None, description="Optional user identifier")
    session_id: Optional[str] = Field(None, description="Optional session identifier")

class PIIData(BaseModel):
    """PII (Personally Identifiable Information) detection results"""
    found: bool
    entities: List[Dict[str, Any]]
    confidence_score: float
    tokenized_data: Dict[str, str]

class PCIData(BaseModel):
    """PCI (Payment Card Industry) detection results"""
    found: bool
    entities: List[Dict[str, Any]]
    confidence_score: float
    tokenized_data: Dict[str, str]

class ExtractionResponse(BaseModel):
    success: bool
    message: str
    processing_time_ms: float
    pii: PIIData
    pci: PCIData
    metadata: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    models_loaded: Dict[str, bool]

# Utility function to convert numpy types to Python native types
def convert_numpy_types(obj):
    """Recursively convert numpy types to Python native types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

# PII/PCI Extractor class with BERT
class PIIPCIExtractor:
    def __init__(self):
        logger.info("Initializing PII/PCI Extractor with BERT models...")
        
        # PCI patterns (Credit Cards, Bank Info)
        self.pci_patterns = {
            'visa': r'4[0-9]{12}(?:[0-9]{3})?',
            'mastercard': r'5[1-5][0-9]{14}',
            'amex': r'3[47][0-9]{13}',
            'discover': r'6(?:011|5[0-9]{2})[0-9]{12}',
            'diners': r'3[0689][0-9]{11}',
            'jcb': r'35[0-9]{14}',
            'generic_cc': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
            'cvv': r'\b\d{3,4}\b',
            'routing_number': r'\b[0-9]{9}\b',
            'account_number': r'\b\d{8,17}\b'
        }
        
        # PII patterns (Personal Information)
        self.pii_patterns = {
            'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b',
            'phone': r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'drivers_license': r'\b[A-Z]{1,2}[0-9]{6,8}\b',
            'passport': r'\b[A-Z]{2}[0-9]{7}\b',
            'date_of_birth': r'\b(?:0[1-9]|1[0-2])[/-](?:0[1-9]|[12][0-9]|3[01])[/-](?:19|20)\d{2}\b',
            'address_pattern': r'\b\d+\s+[A-Za-z\s]+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr|Court|Ct|Place|Pl)\b'
        }
        
        # Initialize BERT models
        self.models_loaded = {}
        self._load_bert_models()
        
    def _load_bert_models(self):
        """Load BERT models for PII/PCI detection"""
        try:
            # 1. Named Entity Recognition Model for PII detection
            logger.info("Loading NER model for PII detection...")
            self.ner_pipeline = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                tokenizer="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="simple"
            )
            self.models_loaded['ner'] = True
            logger.info("✅ NER model loaded")
            
        except Exception as e:
            logger.error(f"❌ Failed to load NER model: {e}")
            self.ner_pipeline = None
            self.models_loaded['ner'] = False
        
        try:
            # 2. PII-specific BERT model (if available, otherwise use general NER)
            logger.info("Loading PII-specific model...")
            # Using a model trained on privacy/PII detection
            self.pii_pipeline = pipeline(
                "ner",
                model="StanfordAIMI/stanford-deidentifier-base",
                aggregation_strategy="simple"
            )
            self.models_loaded['pii_specific'] = True
            logger.info("✅ PII-specific model loaded")
            
        except Exception as e:
            logger.warning(f"⚠️ PII-specific model not available, using general NER: {e}")
            self.pii_pipeline = self.ner_pipeline
            self.models_loaded['pii_specific'] = False
        
        logger.info(f"Models loaded: {self.models_loaded}")
    
    def detect_pci_regex(self, text: str) -> Dict[str, Any]:
        """Detect PCI information using regex patterns"""
        pci_results = {
            'found': False,
            'entities': [],
            'confidence_score': 0.0,
            'raw_data': {}
        }
        
        # Credit Card Detection
        clean_text = text.replace('-', '').replace(' ', '')
        for card_type, pattern in self.pci_patterns.items():
            if card_type in ['cvv', 'routing_number', 'account_number']:
                continue
                
            matches = re.finditer(pattern, clean_text)
            for match in matches:
                card_number = match.group()
                if self._validate_credit_card(card_number):
                    pci_results['entities'].append({
                        'type': 'credit_card',
                        'subtype': card_type,
                        'value': card_number,
                        'confidence': 0.95,
                        'start_pos': match.start(),
                        'end_pos': match.end(),
                        'detection_method': 'regex'
                    })
                    pci_results['raw_data'][f'{card_type}_card'] = card_number
                    pci_results['found'] = True
        
        # CVV Detection (only if credit card found)
        if pci_results['found']:
            cvv_matches = re.finditer(self.pci_patterns['cvv'], text)
            for match in cvv_matches:
                pci_results['entities'].append({
                    'type': 'cvv',
                    'subtype': 'security_code',
                    'value': match.group(),
                    'confidence': 0.85,
                    'start_pos': match.start(),
                    'end_pos': match.end(),
                    'detection_method': 'regex'
                })
                pci_results['raw_data']['cvv'] = match.group()
        
        # Bank Account Information
        routing_matches = re.finditer(self.pci_patterns['routing_number'], text)
        for match in routing_matches:
            pci_results['entities'].append({
                'type': 'bank_routing',
                'subtype': 'routing_number',
                'value': match.group(),
                'confidence': 0.80,
                'start_pos': match.start(),
                'end_pos': match.end(),
                'detection_method': 'regex'
            })
            pci_results['raw_data']['routing_number'] = match.group()
            pci_results['found'] = True
        
        # Calculate overall confidence
        if pci_results['entities']:
            pci_results['confidence_score'] = sum(e['confidence'] for e in pci_results['entities']) / len(pci_results['entities'])
        
        return pci_results
    
    def detect_pii_regex(self, text: str) -> Dict[str, Any]:
        """Detect PII information using regex patterns"""
        pii_results = {
            'found': False,
            'entities': [],
            'confidence_score': 0.0,
            'raw_data': {}
        }
        
        # SSN Detection
        ssn_matches = re.finditer(self.pii_patterns['ssn'], text)
        for match in ssn_matches:
            pii_results['entities'].append({
                'type': 'ssn',
                'subtype': 'social_security_number',
                'value': match.group(),
                'confidence': 0.95,
                'start_pos': match.start(),
                'end_pos': match.end(),
                'detection_method': 'regex'
            })
            pii_results['raw_data']['ssn'] = match.group()
            pii_results['found'] = True
        
        # Phone Number Detection
        phone_matches = re.finditer(self.pii_patterns['phone'], text)
        for match in phone_matches:
            pii_results['entities'].append({
                'type': 'phone',
                'subtype': 'phone_number',
                'value': match.group(),
                'confidence': 0.90,
                'start_pos': match.start(),
                'end_pos': match.end(),
                'detection_method': 'regex'
            })
            pii_results['raw_data']['phone'] = match.group()
            pii_results['found'] = True
        
        # Email Detection
        email_matches = re.finditer(self.pii_patterns['email'], text)
        for match in email_matches:
            pii_results['entities'].append({
                'type': 'email',
                'subtype': 'email_address',
                'value': match.group(),
                'confidence': 0.95,
                'start_pos': match.start(),
                'end_pos': match.end(),
                'detection_method': 'regex'
            })
            pii_results['raw_data']['email'] = match.group()
            pii_results['found'] = True
        
        # Date of Birth Detection
        dob_matches = re.finditer(self.pii_patterns['date_of_birth'], text)
        for match in dob_matches:
            pii_results['entities'].append({
                'type': 'date_of_birth',
                'subtype': 'birth_date',
                'value': match.group(),
                'confidence': 0.85,
                'start_pos': match.start(),
                'end_pos': match.end(),
                'detection_method': 'regex'
            })
            pii_results['raw_data']['date_of_birth'] = match.group()
            pii_results['found'] = True
        
        # Address Detection
        address_matches = re.finditer(self.pii_patterns['address_pattern'], text, re.IGNORECASE)
        for match in address_matches:
            pii_results['entities'].append({
                'type': 'address',
                'subtype': 'street_address',
                'value': match.group(),
                'confidence': 0.75,
                'start_pos': match.start(),
                'end_pos': match.end(),
                'detection_method': 'regex'
            })
            pii_results['raw_data']['address'] = match.group()
            pii_results['found'] = True
        
        # Calculate overall confidence
        if pii_results['entities']:
            pii_results['confidence_score'] = sum(e['confidence'] for e in pii_results['entities']) / len(pii_results['entities'])
        
        return pii_results
    
    def detect_pii_bert(self, text: str) -> Dict[str, Any]:
        """Detect PII using BERT NER models"""
        bert_pii_results = {
            'found': False,
            'entities': [],
            'confidence_score': 0.0,
            'raw_data': {}
        }
        
        if not self.ner_pipeline:
            return bert_pii_results
        
        try:
            # Use PII-specific model if available, otherwise general NER
            pipeline_to_use = self.pii_pipeline if self.pii_pipeline else self.ner_pipeline
            ner_results = pipeline_to_use(text)
            
            for entity in ner_results:
                entity_type = entity['entity_group'].upper()
                confidence = float(entity['score'])  # Convert numpy.float32 to Python float
                
                # Map NER labels to PII categories
                pii_mapping = {
                    'PER': 'person_name',
                    'PERSON': 'person_name',
                    'LOC': 'location',
                    'LOCATION': 'location',
                    'ORG': 'organization',
                    'ORGANIZATION': 'organization',
                    'MISC': 'miscellaneous',
                    'DATE': 'date',
                    'TIME': 'time',
                    'MONEY': 'monetary_value',
                    'PERCENT': 'percentage'
                }
                
                if entity_type in pii_mapping and confidence > 0.7:
                    bert_pii_results['entities'].append({
                        'type': 'pii_entity',
                        'subtype': pii_mapping[entity_type],
                        'value': entity['word'],
                        'confidence': round(confidence, 3),
                        'start_pos': int(entity.get('start', 0)),  # Convert numpy int to Python int
                        'end_pos': int(entity.get('end', 0)),     # Convert numpy int to Python int
                        'detection_method': 'bert_ner'
                    })
                    bert_pii_results['raw_data'][pii_mapping[entity_type]] = entity['word']
                    bert_pii_results['found'] = True
            
            # Calculate overall confidence
            if bert_pii_results['entities']:
                bert_pii_results['confidence_score'] = sum(e['confidence'] for e in bert_pii_results['entities']) / len(bert_pii_results['entities'])
            
        except Exception as e:
            logger.error(f"BERT PII detection failed: {e}")
        
        return bert_pii_results
    
    def _validate_credit_card(self, card_number: str) -> bool:
        """Validate credit card using Luhn algorithm"""
        def luhn_check(card_num):
            def digits_of(n):
                return [int(d) for d in str(n)]
            
            digits = digits_of(card_num)
            odd_digits = digits[-1::-2]
            even_digits = digits[-2::-2]
            checksum = sum(odd_digits)
            for d in even_digits:
                checksum += sum(digits_of(d*2))
            return checksum % 10 == 0
        
        # Remove any spaces or dashes
        clean_number = re.sub(r'[\s-]', '', card_number)
        
        # Check if it's all digits and right length
        if not clean_number.isdigit() or len(clean_number) < 13 or len(clean_number) > 19:
            return False
        
        return luhn_check(clean_number)
    
    def tokenize_sensitive_data(self, data: str, data_type: str) -> str:
        """Tokenize sensitive information"""
        if not data:
            return None
        
        # Create a hash-based token
        hash_object = hashlib.md5(data.encode())
        hash_hex = hash_object.hexdigest()[:8]
        
        return f"TOKEN_{data_type.upper()}_{hash_hex}"
    
    def process(self, text: str) -> Dict[str, Any]:
        """Main processing pipeline for PII/PCI detection"""
        
        # 1. Detect PCI information
        pci_regex_results = self.detect_pci_regex(text)
        
        # 2. Detect PII information (regex)
        pii_regex_results = self.detect_pii_regex(text)
        
        # 3. Detect PII information (BERT)
        pii_bert_results = self.detect_pii_bert(text)
        
        # 4. Combine PII results (regex + BERT)
        combined_pii_entities = pii_regex_results['entities'] + pii_bert_results['entities']
        combined_pii_raw_data = {**pii_regex_results['raw_data'], **pii_bert_results['raw_data']}
        
        # Remove duplicates based on value and type
        seen = set()
        unique_pii_entities = []
        for entity in combined_pii_entities:
            key = (entity['value'], entity['type'])
            if key not in seen:
                seen.add(key)
                unique_pii_entities.append(entity)
        
        # 5. Tokenize sensitive data
        pci_tokenized = {}
        for key, value in pci_regex_results['raw_data'].items():
            pci_tokenized[key] = self.tokenize_sensitive_data(value, f"PCI_{key}")
        
        pii_tokenized = {}
        for key, value in combined_pii_raw_data.items():
            pii_tokenized[key] = self.tokenize_sensitive_data(value, f"PII_{key}")
        
        # 6. Calculate combined confidence scores
        pii_confidence = 0.0
        if unique_pii_entities:
            pii_confidence = sum(e['confidence'] for e in unique_pii_entities) / len(unique_pii_entities)
        
        # 7. Build final response - Convert all numpy types to Python native types
        result = {
            'pii': {
                'found': len(unique_pii_entities) > 0,
                'entities': convert_numpy_types(unique_pii_entities),
                'confidence_score': round(float(pii_confidence), 3),
                'tokenized_data': pii_tokenized
            },
            'pci': {
                'found': pci_regex_results['found'],
                'entities': convert_numpy_types(pci_regex_results['entities']),
                'confidence_score': round(float(pci_regex_results['confidence_score']), 3),
                'tokenized_data': pci_tokenized
            },
            'metadata': {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'processing_methods': ['regex', 'bert_ner'],
                'models_used': {
                    'ner': self.models_loaded.get('ner', False),
                    'pii_specific': self.models_loaded.get('pii_specific', False)
                },
                'total_entities_found': len(unique_pii_entities) + len(pci_regex_results['entities']),
                'risk_level': self._calculate_risk_level(len(unique_pii_entities), len(pci_regex_results['entities']))
            }
        }
        
        return result
    
    def _calculate_risk_level(self, pii_count: int, pci_count: int) -> str:
        """Calculate risk level based on detected entities"""
        total_sensitive = pii_count + pci_count
        
        if total_sensitive == 0:
            return "LOW"
        elif total_sensitive <= 2:
            return "MEDIUM"
        elif total_sensitive <= 4:
            return "HIGH"
        else:
            return "CRITICAL"

# Initialize FastAPI app
app = FastAPI(
    title="PII/PCI Detection API with BERT",
    description="Detect Personally Identifiable Information (PII) and Payment Card Industry (PCI) data using RegEx + BERT NER",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Initialize extractor
logger.info("Starting PII/PCI Detection API...")
extractor = PIIPCIExtractor()

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat() + 'Z',
        version="1.0.0",
        models_loaded=extractor.models_loaded
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat() + 'Z',
        version="1.0.0",
        models_loaded=extractor.models_loaded
    )

@app.post("/extract", response_model=ExtractionResponse)
async def extract_pii_pci(message: ChatMessage):
    """
    Extract PII and PCI information from text
    
    - **text**: Text to analyze for PII/PCI data
    - **user_id**: Optional user identifier
    - **session_id**: Optional session identifier
    
    Returns separate objects for PII and PCI findings
    """
    try:
        import time
        start_time = time.time()
        
        # Process the message
        result = extractor.process(message.text)
        
        # Add request metadata
        if message.user_id or message.session_id:
            result['metadata']['request_info'] = {
                'user_id': message.user_id,
                'session_id': message.session_id
            }
        
        processing_time = (time.time() - start_time) * 1000
        
        return ExtractionResponse(
            success=True,
            message="PII/PCI extraction completed successfully",
            processing_time_ms=round(processing_time, 2),
            pii=PIIData(**result['pii']),
            pci=PCIData(**result['pci']),
            metadata=result['metadata']
        )
        
    except Exception as e:
        logger.error(f"PII/PCI extraction failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Extraction failed: {str(e)}"
        )

@app.post("/scan")
async def quick_scan(message: ChatMessage):
    """
    Quick scan to check if text contains PII or PCI data (summary only)
    
    - **text**: Text to scan
    """
    try:
        import time
        start_time = time.time()
        
        result = extractor.process(message.text)
        processing_time = (time.time() - start_time) * 1000
        
        return {
            "success": True,
            "message": "Quick scan completed",
            "processing_time_ms": round(processing_time, 2),
            "summary": {
                "pii_found": result['pii']['found'],
                "pci_found": result['pci']['found'],
                "pii_entity_count": len(result['pii']['entities']),
                "pci_entity_count": len(result['pci']['entities']),
                "risk_level": result['metadata']['risk_level'],
                "confidence_scores": {
                    "pii": result['pii']['confidence_score'],
                    "pci": result['pci']['confidence_score']
                }
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Quick scan failed: {str(e)}"
        )

@app.get("/models")
async def get_model_status():
    """Get status of loaded BERT models"""
    return {
        "models_loaded": extractor.models_loaded,
        "model_details": {
            "ner": "dbmdz/bert-large-cased-finetuned-conll03-english",
            "pii_specific": "StanfordAIMI/stanford-deidentifier-base"
        },
        "supported_pii_types": [
            "person_name", "location", "organization", "date", "time", 
            "ssn", "phone", "email", "address", "date_of_birth"
        ],
        "supported_pci_types": [
            "credit_card", "cvv", "bank_routing", "account_number"
        ]
    }

@app.get("/stats")
async def get_stats():
    """Get API statistics and capabilities"""
    return {
        "api_version": "1.0.0",
        "purpose": "PII/PCI Detection",
        "detection_methods": ["regex", "bert_ner"],
        "supported_pii_entities": [
            "Social Security Number", "Phone Number", "Email Address",
            "Street Address", "Date of Birth", "Person Names", "Locations", "Organizations"
        ],
        "supported_pci_entities": [
            "Visa", "Mastercard", "American Express", "Discover", 
            "Diners Club", "JCB", "CVV", "Bank Routing Number"
        ],
        "risk_levels": ["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        "features": {
            "luhn_validation": True,
            "tokenization": True,
            "confidence_scoring": True,
            "bert_ner": extractor.models_loaded.get('ner', False),
            "pii_specific_model": extractor.models_loaded.get('pii_specific', False)
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "Nerbertpiipci:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
