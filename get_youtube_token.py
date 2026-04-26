"""
get_youtube_token.py
━━━━━━━━━━━━━━━━━━━━
Run this ONCE locally (outside Docker) to obtain your YouTube OAuth2
refresh token. Paste the output values into your .env file.

Usage:
    pip install google-auth-oauthlib
    python get_youtube_token.py

Pre-requisites:
  1. Go to https://console.cloud.google.com
  2. Create a project → Enable "YouTube Data API v3"
  3. Create OAuth2 credentials (type: Desktop App)
  4. Download the client_secrets JSON file
  5. Set YOUTUBE_CLIENT_ID and YOUTUBE_CLIENT_SECRET below
"""

import os
import json

CLIENT_ID     = os.getenv("YOUTUBE_CLIENT_ID", "YOUR_CLIENT_ID_HERE")
CLIENT_SECRET = os.getenv("YOUTUBE_CLIENT_SECRET", "YOUR_CLIENT_SECRET_HERE")

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

def main():
    from google_auth_oauthlib.flow import InstalledAppFlow

    client_config = {
        "installed": {
            "client_id":     CLIENT_ID,
            "client_secret": CLIENT_SECRET,
            "redirect_uris": ["urn:ietf:wg:oauth:2.0:oob", "http://localhost"],
            "auth_uri":      "https://accounts.google.com/o/oauth2/auth",
            "token_uri":     "https://oauth2.googleapis.com/token",
        }
    }

    flow = InstalledAppFlow.from_client_config(client_config, SCOPES)
    creds = flow.run_local_server(port=0)

    print("\n" + "=" * 60)
    print("  Copy these values into your .env file:")
    print("=" * 60)
    print(f"  YOUTUBE_REFRESH_TOKEN={creds.refresh_token}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
