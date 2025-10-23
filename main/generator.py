import requests
import json
import os
import uuid
from datetime import datetime, timedelta
import csv
import random

# Local API endpoint (ensure your local LLM server is running)
API_URL = "http://127.0.0.1:1234/v1/completions"

# Set headers
headers = {
    "Content-Type": "application/json"
}

# Define order data types for training/testing

order_data_types = [
    {
        "type": "E-commerce Order",
        "category": "Online Retail",
        "description": "Standard online purchase with customer details and payment info",
        "example": "Amazon-style order with shipping address and credit card"
    },
    {
        "type": "Restaurant Order",
        "category": "Food Service",
        "description": "Food delivery order with customer contact and payment details",
        "example": "DoorDash/UberEats style order with delivery info"
    },
    {
        "type": "Subscription Service",
        "category": "Recurring Payment",
        "description": "Monthly/yearly subscription with auto-renewal payment info",
        "example": "Netflix/Spotify style subscription with billing details"
    },
    {
        "type": "Hotel Booking",
        "category": "Travel",
        "description": "Hotel reservation with guest details and payment information",
        "example": "Booking.com style reservation with guest and card info"
    },
    {
        "type": "Flight Booking",
        "category": "Travel",
        "description": "Airline ticket purchase with passenger and payment details",
        "example": "Airline booking with passenger info and credit card"
    },
    {
        "type": "Medical Appointment",
        "category": "Healthcare",
        "description": "Medical service booking with patient info and insurance/payment",
        "example": "Doctor appointment with patient details and payment method"
    },
    {
        "type": "Insurance Purchase",
        "category": "Financial Services",
        "description": "Insurance policy purchase with personal and payment information",
        "example": "Auto/health insurance with personal details and payment"
    },
    {
        "type": "Banking Transaction",
        "category": "Financial Services",
        "description": "Bank transfer or payment with account and personal details",
        "example": "Wire transfer or online banking transaction"
    }
]

# Directories to save output
OUTPUT_DIR_TXT = "generated_orders_txt"
OUTPUT_DIR_JSON = "generated_orders_json"
SUMMARY_CSV = "order_training_summary.csv"

os.makedirs(OUTPUT_DIR_TXT, exist_ok=True)
os.makedirs(OUTPUT_DIR_JSON, exist_ok=True)

# Prepare summary CSV
summary_fieldnames = [
    "id", "filename_txt", "filename_json", "order_type", "category",
    "description", "example_hint", "prompt_tokens",
    "generated_tokens", "temperature", "top_p", "model", "timestamp",
    "raw_order_data", "contains_pii", "contains_pci"
]

# Write header only if CSV doesn't exist
if not os.path.exists(SUMMARY_CSV):
    with open(SUMMARY_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()

# Sample data for realistic generation
sample_names = ["John Smith", "Sarah Johnson", "Michael Brown", "Emily Davis", "David Wilson", "Lisa Anderson", "Robert Taylor", "Jennifer Martinez", "William Garcia", "Amanda Rodriguez"]
sample_emails = ["@gmail.com", "@yahoo.com", "@hotmail.com", "@outlook.com", "@company.com"]
sample_cities = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose"]
sample_states = ["NY", "CA", "IL", "TX", "AZ", "PA", "FL", "OH", "GA", "NC"]

def generate_fake_credit_card():
    """Generate fake but realistic credit card numbers for testing"""
    prefixes = ["4111", "4532", "5555", "5105", "3782", "3714"]  # Test card prefixes
    prefix = random.choice(prefixes)
    remaining = ''.join([str(random.randint(0, 9)) for _ in range(12)])
    return prefix + remaining

def generate_fake_ssn():
    """Generate fake SSN for testing (using invalid ranges)"""
    return f"999-{random.randint(10, 99)}-{random.randint(1000, 9999)}"

# Outer loop: iterate over each order type
for order_data in order_data_types:
    order_type = order_data['type']
    description = order_data['description']
    category = order_data['category']
    example_hint = order_data.get('example', 'a typical scenario')

    print(f"\n🚀 Generating 100 sample orders for: {order_type}...")

    # Inner loop: generate 100 unique orders for this type
    for i in range(100):
        # Generate unique ID for this instance
        unique_id = str(uuid.uuid4())[:8]  # Short UUID
        timestamp = datetime.now().isoformat()
        
        # Generate random realistic data
        customer_name = random.choice(sample_names)
        customer_email = customer_name.lower().replace(" ", ".") + random.choice(sample_emails)
        phone = f"({random.randint(200, 999)}) {random.randint(200, 999)}-{random.randint(1000, 9999)}"
        address = f"{random.randint(100, 9999)} {random.choice(['Main St', 'Oak Ave', 'Park Blvd', 'First St', 'Second Ave'])}"
        city = random.choice(sample_cities)
        state = random.choice(sample_states)
        zipcode = f"{random.randint(10000, 99999)}"
        credit_card = generate_fake_credit_card()
        ssn = generate_fake_ssn()
        order_amount = round(random.uniform(25.99, 999.99), 2)
        order_date = (datetime.now() - timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d")

        # Craft the prompt
        prompt = f"""
Generate a realistic but fictional {order_type.lower()} record for training/testing purposes.

Context:
- Category: {category}
- Description: {description}
- Example: {example_hint}

Include the following realistic dummy data:
- Customer Name: {customer_name}
- Email: {customer_email}
- Phone: {phone}
- Address: {address}, {city}, {state} {zipcode}
- Credit Card: {credit_card}
- SSN: {ssn}
- Order Amount: ${order_amount}
- Order Date: {order_date}

Requirements:
- Create a structured order record with all relevant fields
- Include order details specific to the {order_type.lower()}
- Add realistic product/service details
- Include timestamps, order IDs, and transaction details
- Make it look like real business data
- Include both PII (personal info) and PCI (payment card info)
- Format as a realistic business record/receipt

Generate a complete, realistic order record:
"""

        # Payload for the local LLM server
        payload = {
            "model": "qwen/qwen3-coder-30b",  # Update to your actual model
            "prompt": prompt.strip(),
            "max_tokens": 400,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": False
        }

        # Send POST request
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)

            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('choices', [{}])[0].get('text', '').strip()

                # Extract token usage (if available)
                usage = result.get('usage', {})
                prompt_tokens = usage.get('prompt_tokens', 'unknown')
                generated_tokens = usage.get('completion_tokens', 'unknown')

                # Generate unique filenames using counter i
                safe_type = order_type.replace(' ', '_').replace('/', '_').replace('\\', '_')
                filename_txt = f"{safe_type}_{unique_id}_{i:03d}.txt"
                filename_json = f"{safe_type}_{unique_id}_{i:03d}.json"

                txt_path = os.path.join(OUTPUT_DIR_TXT, filename_txt)
                json_path = os.path.join(OUTPUT_DIR_JSON, filename_json)

                # Save raw order data to TXT
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(f"=== Generated Order Data ===\n")
                    f.write(f"ID: {unique_id}\n")
                    f.write(f"Type: {order_type}\n")
                    f.write(f"Iteration: {i+1}/100\n")
                    f.write(f"Category: {category}\n")
                    f.write(f"Generated on: {timestamp}\n")
                    f.write(f"Contains PII: Yes (Name, Email, Phone, Address, SSN)\n")
                    f.write(f"Contains PCI: Yes (Credit Card Number)\n\n")
                    f.write("=== ORDER DETAILS ===\n")
                    f.write(generated_text)

                # Save structured data to JSON
                structured_data = {
                    "id": unique_id,
                    "iteration": i + 1,
                    "order_type": order_type,
                    "category": category,
                    "description": description,
                    "example_hint": example_hint,
                    "prompt": prompt.strip(),
                    "raw_order_data": generated_text,
                    "model": payload["model"],
                    "temperature": payload["temperature"],
                    "top_p": payload["top_p"],
                    "max_tokens": payload["max_tokens"],
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": generated_tokens,
                    "timestamp": timestamp,
                    "contains_pii": True,
                    "contains_pci": True,
                    "dummy_data": {
                        "customer_name": customer_name,
                        "customer_email": customer_email,
                        "phone": phone,
                        "address": f"{address}, {city}, {state} {zipcode}",
                        "credit_card": credit_card,
                        "ssn": ssn,
                        "order_amount": order_amount,
                        "order_date": order_date
                    }
                }

                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(structured_data, f, indent=2, ensure_ascii=False)

                # Append summary row to CSV
                summary_row = {
                    "id": unique_id,
                    "filename_txt": filename_txt,
                    "filename_json": filename_json,
                    "order_type": order_type,
                    "category": category,
                    "description": description,
                    "example_hint": example_hint,
                    "prompt_tokens": prompt_tokens,
                    "generated_tokens": generated_tokens,
                    "temperature": payload["temperature"],
                    "top_p": payload["top_p"],
                    "model": payload["model"],
                    "timestamp": timestamp,
                    "raw_order_data": generated_text,
                    "contains_pii": "Yes",
                    "contains_pci": "Yes"
                }

                with open(SUMMARY_CSV, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
                    writer.writerow(summary_row)

                print(f"✅ [{order_type}] Order {i+1}/100 saved: TXT → {filename_txt}, JSON → {filename_json}")

            else:
                print(f"❌ [{order_type}] Error {response.status_code} on order {i+1}: {response.text}")

        except requests.exceptions.RequestException as e:
            print(f"⚠️ [{order_type}] Request failed for order {i+1}: {str(e)}")
        except Exception as e:
            print(f"⚠️ [{order_type}] Unexpected error for order {i+1}: {str(e)}")

print("\n🎉 All order data generated! 100 per order type.")
print(f"📊 Summary saved to: {SUMMARY_CSV}")
print(f"📁 Text files saved to: {OUTPUT_DIR_TXT}/")
print(f"📁 JSON files saved to: {OUTPUT_DIR_JSON}/")
print("\n⚠️  IMPORTANT: This data contains dummy PII and PCI information for testing purposes only!")
print("   - All credit card numbers use test prefixes")
print("   - All SSNs use invalid ranges (999-xx-xxxx)")
print("   - All personal information is fictional")
