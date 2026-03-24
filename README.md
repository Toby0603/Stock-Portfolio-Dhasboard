# Stock Tracker

A GitHub-ready Streamlit app that:
- downloads 3 years of Yahoo Finance data per ticker
- builds an XGBoost classification model per ticker
- estimates probability of an upward move
- reports model metrics and a ranking score

## Files
- `app.py` — main Streamlit app
- `requirements.txt` — Python dependencies
- `.streamlit/config.toml` — basic Streamlit config

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Create a new GitHub repository.
2. Upload these files to the repo root:
   - `app.py`
   - `requirements.txt`
   - `.streamlit/config.toml`
3. Push to GitHub.
4. In Streamlit Community Cloud, create a new app from that repo.
5. Set the entrypoint to `app.py`.
6. Deploy.

## Notes
- Internet connection is required.
- Tickers for London stocks often need the `.L` suffix.
- This app is for educational and research use only and is not financial advice.
