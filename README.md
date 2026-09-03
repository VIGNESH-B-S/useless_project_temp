<img width="1280" height="640" alt="git (1)" src="https://github.com/user-attachments/assets/8920b256-2ba8-4988-b824-5351134eb4bd" />



# Kuthivara_Meter🎯


## Simple open-cv based software that tracks random scribbling in a paper
### Team Name: Brains


### Team Members
- Team Lead: Vignesh BS - Model Engineering College
- Member 2: Navaneeth J Menon- Model Engineering College


### Project Description
Open-cv based software that tracks the amount of ink used while scribbling

### The Problem (that doesn't exist)
How much ink do you waste while scribbling.

### The Solution (that nobody asked for)
[How are you solving it? Keep it fun!]

## Technical Details
### Technologies/Components Used
For Software:
- Python
- Open-cv
- Html
- Css
-Javascript


json
{
  "success": true,
  "scribble_length_mm": 389.61,
  "scribble_length_cm": 38.96,
  "scribble_length_m": 0.39,
  "estimated_ink_ml": 0.0001948,
  "estimated_ink_microlitres": 0.1948,
  "confidence": 0.92
}


### GET /health

Deployment health check:

json
{
  "service": "KUTHIVARA Meter API",
  "status": "ok"
}


## Deployment

### Railway — Flask/OpenCV backend

1. Push the repository to GitHub.
2. Create a Railway project from the repository.
3. Set Railway *Root Directory* to:

text
/KUTHIVARA_METER/backend


4. Railway uses railway.json to run:

text
gunicorn --bind 0.0.0.0:$PORT app:app


5. Generate a public Railway domain.
6. Confirm:

text
https://YOUR-RAILWAY-DOMAIN/health


### Vercel — static frontend

1. Import the same GitHub repository into Vercel.
2. Set *Root Directory* to:

text
KUTHIVARA_METER


3. Choose framework preset *Other*.
4. Leave Build Command and Output Directory empty.
5. In frontend/config.js, set:

js
window.KUTHIVARA_API_URL = "https://YOUR-RAILWAY-DOMAIN";


6. Commit and push config.js; Vercel will redeploy.

No Vercel environment variable is required in the current MVP because the public Railway URL is held in frontend/config.js.

# Screenshots (Add at least 3)
<img width="1902" height="952" alt="image" src="https://github.com/user-attachments/assets/f280b708-3478-415d-8f40-0daf2501cb81" />


<img width="1917" height="972" alt="image" src="https://github.com/user-attachments/assets/6791e1bb-bd97-4c48-85ae-bf19de2234c9" />


![Uploading image.png…]()





### Project Demo
# Video



https://github.com/user-attachments/assets/b1ced352-3f17-457d-b5b0-00a3959915ee




  

---
Made with ❤️ at TinkerHub Useless Projects 

![Static Badge](https://img.shields.io/badge/TinkerHub-24?color=%23000000&link=https%3A%2F%2Fwww.tinkerhub.org%2F)
![Static Badge](https://img.shields.io/badge/UselessProjects--26-26?link=https%3A%2F%2Ftinkerhub.org%2Fevents%2F1M8ORET9A1%2Fuseless-projects-3.0)



