# Security
- Never commit `.env` or API keys. Keys are read from the environment only and are never logged or returned by the API.
- Set `API_KEY` and a strict `CORS_ORIGINS` before exposing the service publicly.
- The website stores an optional Groq or Gemini key in your browser's localStorage and sends it only to Groq or Google.
- Report vulnerabilities by opening a private security advisory on GitHub.
