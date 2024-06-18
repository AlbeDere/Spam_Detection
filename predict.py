import imaplib
import email
from email.header import decode_header
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch
import re
import requirements

# Function to translate text to English using Google Translate API
def translate_to_english(text, src_lang):
    from googletrans import Translator
    translator = Translator()
    translation = translator.translate(text, src=src_lang, dest='en')
    return translation.text

# Function to classify email content as spam or not spam
def classify_spam(email_content):
    # Load the trained model and tokenizer
    model_path = 'C:/Users/Praxis/Documents/trained_model'  # Update with your actual path
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)

    # Move the model to the appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Tokenize the input text
    encoded_input = tokenizer(email_content, truncation=True, padding=True, return_tensors='pt').to(device)

    # Perform the classification
    with torch.no_grad():
        outputs = model(**encoded_input)
        logits = outputs.logits
        predicted_label = logits.argmax(dim=-1).item()

    return predicted_label

# Gmail IMAP server details
IMAP_SERVER = 'imap.gmail.com'
IMAP_PORT = 993

# Your Gmail credentials
EMAIL = requirements.EMAIL
PASSWORD = requirements.PASSWORD

# Connect to Gmail IMAP server
mail = imaplib.IMAP4_SSL(IMAP_SERVER, IMAP_PORT)

# Login to your account
mail.login(EMAIL, PASSWORD)

# Select the mailbox you want to access (e.g., 'inbox')
mail.select('inbox')

# Search for unread emails in the mailbox
result, data = mail.search(None, 'UNSEEN')

# Iterate through all unread emails
for num in data[0].split():
    # Fetch the email data
    result, data = mail.fetch(num, '(RFC822)')
    
    # Parse the email using the email library
    raw_email = data[0][1]
    msg = email.message_from_bytes(raw_email)
    
    # Get the subject of the email
    subject = decode_header(msg['Subject'])[0][0]
    if isinstance(subject, bytes):
        subject = subject.decode()
    
    # Get the sender of the email
    sender = decode_header(msg['From'])[0][0]
    if isinstance(sender, bytes):
        sender = sender.decode()
    
    print('Subject:', subject)
    print('From:', sender)
    
    # Fetch the email content
    email_content = ''
    for part in msg.walk():
        if part.get_content_type() == 'text/plain':
            content = part.get_payload(decode=True).decode()
            email_content += content
    
    # Translate the email content to English
    translated_content = translate_to_english(email_content, 'auto')  # Automatically detect language
    
    # Clean the translated content (remove non-alphanumeric characters)
    cleaned_content = re.sub(r'\W+', ' ', translated_content)
    
    # Classify the content as spam or not spam
    predicted_label = classify_spam(cleaned_content)
    
    # Print classification result
    if predicted_label == 1:
        print('Spam')
        mail.store(num, '+X-GM-LABELS', '"Potential Spam"')
    else:
        print('Not Spam')
    
    # Mark the email as unread
    mail.store(num, '-FLAGS', '\\Seen')

# Close the connection
mail.close()
mail.logout()
