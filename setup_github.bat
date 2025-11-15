@echo off
echo Setting up GitHub repository for EPL Predictor...

git init
git add .
git commit -m "Initial commit - EPL Points Prediction ML Web App"

echo.
echo Next steps:
echo 1. Create a new repository on GitHub.com named 'epl-predictor'
echo 2. Run these commands:
echo    git remote add origin https://github.com/YOUR_USERNAME/epl-predictor.git
echo    git branch -M main
echo    git push -u origin main
echo.
echo 3. Then deploy on Render.com using the GitHub repository
echo.
pause