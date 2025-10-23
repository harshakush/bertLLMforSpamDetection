# Create corrected PII/PCI client
import requests
import json
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class APIResponse:
    """Wrapper for API responses"""
    success: bool
    status_code: int
    data: Dict[str, Any]
    response_time_ms: float

class PIIPCIExtractorClient:
    """Python client for PII/PCI Detection API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> APIResponse:
        """Make HTTP request and return standardized response"""
        url = f"{self.base_url}{endpoint}"
        start_time = time.time()
        
        try:
            if method.upper() == 'GET':
                response = self.session.get(url)
            elif method.upper() == 'POST':
                response = self.session.post(url, json=data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            response_time = (time.time() - start_time) * 1000
            
            return APIResponse(
                success=response.status_code == 200,
                status_code=response.status_code,
                data=response.json() if response.content else {},
                response_time_ms=round(response_time, 2)
            )
            
        except requests.exceptions.RequestException as e:
            response_time = (time.time() - start_time) * 1000
            return APIResponse(
                success=False,
                status_code=0,
                data={"error": str(e)},
                response_time_ms=round(response_time, 2)
            )
    
    def health_check(self) -> APIResponse:
        """Check if the API is healthy"""
        return self._make_request('GET', '/health')
    
    def get_root(self) -> APIResponse:
        """Get root endpoint info"""
        return self._make_request('GET', '/')
    
    def extract_pii_pci(self, text: str, user_id: Optional[str] = None, 
                       session_id: Optional[str] = None) -> APIResponse:
        """
        Full PII/PCI extraction with detailed results
        
        Args:
            text: Chat message text to analyze
            user_id: Optional user identifier
            session_id: Optional session identifier
        """
        payload = {"text": text}
        if user_id:
            payload["user_id"] = user_id
        if session_id:
            payload["session_id"] = session_id
            
        return self._make_request('POST', '/extract', payload)
    
    def quick_scan(self, text: str) -> APIResponse:
        """
        Quick scan for PII/PCI data (summary only)
        
        Args:
            text: Chat message text to analyze
        """
        payload = {"text": text}
        return self._make_request('POST', '/scan', payload)
    
    def get_stats(self) -> APIResponse:
        """Get API statistics and supported features"""
        return self._make_request('GET', '/stats')

def demo_usage():
    """Demonstrate usage of all API endpoints"""
    
    # Initialize client
    client = PIIPCIExtractorClient()
    
    print("🚀 PII/PCI Detection API Client Demo")
    print("=" * 50)
    
    # 1. Health Check
    print("\\n1. Health Check:")
    health = client.health_check()
    print(f"Status: {health.status_code}")
    print(f"Response: {json.dumps(health.data, indent=2)}")
    print(f"Response Time: {health.response_time_ms}ms")
    
    if not health.success:
        print("❌ API is not healthy. Exiting...")
        return
    
    # 2. Root endpoint
    print("\\n2. Root Endpoint:")
    root = client.get_root()
    print(f"Response: {json.dumps(root.data, indent=2)}")
    
    # 3. API Stats
    print("\\n3. API Statistics:")
    stats = client.get_stats()
    print(f"Response: {json.dumps(stats.data, indent=2)}")
    
    # 4. Test messages for extraction
    test_messages = [
        {
            "text": "Hi, I'm John Smith and my SSN is 123-45-6789. My credit card 4532123456789012 was charged twice!",
            "description": "Message with name, SSN, and credit card"
        },
        {
            "text": "My email is john.doe@example.com and phone is 555-123-4567",
            "description": "Message with email and phone"
        },
        {
            "text": "I live at 123 Main Street, New York 10001",
            "description": "Message with address and zip code"
        },
        {
            "text": "My Mastercard 5555444433332222 expires next month",
            "description": "Message with Mastercard"
        },
        {
            "text": "Just saying hello, how are you today?",
            "description": "Simple message with no sensitive data"
        }
    ]
    
    # 5. Quick Scan Tests
    print("\\n4. Quick Scan Tests:")
    print("-" * 40)
    
    for i, msg in enumerate(test_messages, 1):
        print(f"\\nTest {i}: {msg['description']}")
        print(f"Message: '{msg['text']}'")
        
        scan_result = client.quick_scan(msg['text'])
        if scan_result.success:
            summary = scan_result.data.get('summary', {})
            print(f"✅ PII Found: {summary.get('pii_found', False)}")
            print(f"   PCI Found: {summary.get('pci_found', False)}")
            print(f"   PII Count: {summary.get('pii_entity_count', 0)}")
            print(f"   PCI Count: {summary.get('pci_entity_count', 0)}")
            print(f"   Risk Level: {summary.get('risk_level', 'LOW')}")
            print(f"   Processing Time: {scan_result.data.get('processing_time_ms', 0)}ms")
        else:
            print(f"❌ Error: {scan_result.data}")
    
    # 6. Full Extraction Tests
    print("\\n\\n5. Full Extraction Tests:")
    print("-" * 45)
    
    for i, msg in enumerate(test_messages, 1):
        print(f"\\nTest {i}: {msg['description']}")
        print(f"Message: '{msg['text']}'")
        
        extract_result = client.extract_pii_pci(
            text=msg['text'],
            user_id=f"user_{i}",
            session_id=f"session_{i}"
        )
        
        if extract_result.success:
            print(f"✅ Processing Time: {extract_result.data.get('processing_time_ms', 0)}ms")
            
            # PII Results
            pii = extract_result.data.get('pii', {})
            print(f"   PII Found: {pii.get('found', False)}")
            if pii.get('entities'):
                print(f"   PII Entities: {len(pii['entities'])}")
                for entity in pii['entities'][:3]:  # Show first 3
                    print(f"     - {entity['type']}: {entity['value']} (confidence: {entity['confidence']})")
            print(f"   PII Confidence: {pii.get('confidence_score', 0)}")
            
            # PCI Results
            pci = extract_result.data.get('pci', {})
            print(f"   PCI Found: {pci.get('found', False)}")
            if pci.get('entities'):
                print(f"   PCI Entities: {len(pci['entities'])}")
                for entity in pci['entities'][:3]:  # Show first 3
                    print(f"     - {entity['type']}: {entity.get('subtype', 'N/A')} (confidence: {entity['confidence']})")
            print(f"   PCI Confidence: {pci.get('confidence_score', 0)}")
            
            # Risk Level
            metadata = extract_result.data.get('metadata', {})
            print(f"   Risk Level: {metadata.get('risk_level', 'LOW')}")
            
        else:
            print(f"❌ Error: {extract_result.data}")
        
        print("-" * 30)

def interactive_mode():
    """Interactive mode for testing custom messages"""
    client = PIIPCIExtractorClient()
    
    print("\\n🔧 Interactive Mode")
    print("Enter chat messages to analyze (type 'quit' to exit)")
    print("Commands: 'scan' for quick scan, 'extract' for full extraction")
    print("-" * 50)
    
    while True:
        try:
            message = input("\\nEnter message: ").strip()
            if message.lower() in ['quit', 'exit', 'q']:
                break
            
            if not message:
                continue
            
            mode = input("Mode (scan/extract) [extract]: ").strip().lower()
            if not mode:
                mode = 'extract'
            
            if mode == 'scan':
                result = client.quick_scan(message)
                if result.success:
                    summary = result.data.get('summary', {})
                    print(f"\\n📊 Quick Scan Results:")
                    print(f"PII Found: {summary.get('pii_found', False)}")
                    print(f"PCI Found: {summary.get('pci_found', False)}")
                    print(f"Risk Level: {summary.get('risk_level', 'LOW')}")
                    print(f"Processing Time: {result.data.get('processing_time_ms', 0)}ms")
                else:
                    print(f"❌ Error: {result.data}")
            
            elif mode == 'extract':
                result = client.extract_pii_pci(message, user_id="interactive_user")
                if result.success:
                    print(f"\\n📋 Full Extraction Results:")
                    print(json.dumps(result.data, indent=2))
                else:
                    print(f"❌ Error: {result.data}")
            
            else:
                print("Invalid mode. Use 'scan' or 'extract'")
                
        except KeyboardInterrupt:
            break
    
    print("\\n👋 Goodbye!")

if __name__ == "__main__":
    print("PII/PCI Detection API Client")
    print("Choose mode:")
    print("1. Demo (automated tests)")
    print("2. Interactive (manual testing)")
    
    choice = input("Enter choice (1/2): ").strip()
    
    if choice == "1":
        demo_usage()
    elif choice == "2":
        interactive_mode()
    else:
        print("Running demo by default...")
        demo_usage()
