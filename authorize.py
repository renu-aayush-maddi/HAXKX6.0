import os
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle

SCOPES = ['https://www.googleapis.com/auth/drive']

def main():
    creds = None
    if os.path.exists('token.json'):
        print("token.json already exists.")
        return
    flow = InstalledAppFlow.from_client_secrets_file(
        'auth/google_creds.json', SCOPES)
    creds = flow.run_local_server(port=0)
    with open('token.json', 'w') as token:
        token.write(creds.to_json())
    print("Authorization complete. token.json saved.")

if __name__ == '__main__':
    main()
