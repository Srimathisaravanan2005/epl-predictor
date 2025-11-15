"""
Flask Web Application for EPL Points Prediction
"""

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
import openai
from src.utils import (load_models, predict_match_result, get_team_stats, 
                      get_upcoming_matches, get_live_matches, calculate_prediction_accuracy,
                      validate_match_input, get_feature_descriptions, convert_numpy_types)
from src.json_utils import safe_jsonify, create_prediction_response, create_error_response

def create_app():
    app = Flask(__name__, template_folder='../templates', static_folder='../static')
    app.secret_key = 'epl_prediction_secret_key_2024'
    
    # Configure OpenAI (set your API key)
    openai_api_key = os.getenv('OPENAI_API_KEY', 'your-openai-api-key-here')
    
    # Flask-Login setup
    login_manager = LoginManager()
    login_manager.init_app(app)
    login_manager.login_view = 'login'
    
    # Context processor for template variables
    @app.context_processor
    def inject_template_vars():
        return {
            'current_year': datetime.now().year
        }
    
    # Simple user class for demo
    class User(UserMixin):
        def __init__(self, id):
            self.id = id
    
    @login_manager.user_loader
    def load_user(user_id):
        return User(user_id)
    
    # Routes
    @app.route('/')
    def home():
        return render_template('home.html')
    
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']
            
            # Simple authentication (in production, use proper password hashing)
            if username and password:  # Accept any non-empty credentials for demo
                user = User(username)
                login_user(user)
                return redirect(url_for('dashboard'))
            else:
                flash('Invalid credentials')
        
        return render_template('login.html')
    
    @app.route('/register', methods=['GET', 'POST'])
    def register():
        if request.method == 'POST':
            username = request.form['username']
            password = request.form['password']
            
            if username and password:
                flash('Registration successful! Please login.')
                return redirect(url_for('login'))
            else:
                flash('Please fill all fields')
        
        return render_template('register.html')
    
    @app.route('/logout')
    @login_required
    def logout():
        logout_user()
        return redirect(url_for('home'))
    
    @app.route('/scoresight')
    @login_required
    def scoresight():
        live_matches = get_live_matches()
        upcoming_matches = get_upcoming_matches()
        accuracy_stats = calculate_prediction_accuracy()
        
        return render_template('scoresight.html', 
                             live_matches=live_matches,
                             upcoming_matches=upcoming_matches,
                             accuracy_stats=accuracy_stats)
    
    @app.route('/predict', methods=['GET', 'POST'])
    @login_required
    def predict():
        if request.method == 'POST':
            try:
                # Get team selection
                home_team = request.form.get('home_team')
                away_team = request.form.get('away_team')
                
                if not home_team or not away_team:
                    return jsonify({'success': False, 'error': 'Please select both teams'})
                
                # Get form data
                match_data = {}
                for field in ['HS', 'AS', 'HST', 'AST', 'HC', 'AC', 'HF', 'AF', 'HY', 'AY', 'HR', 'AR']:
                    match_data[field] = float(request.form.get(field, 0))
                
                # Half-time data
                match_data['HTHG'] = float(request.form.get('HTHG', 0))
                match_data['HTAG'] = float(request.form.get('HTAG', 0))
                match_data['HTR'] = request.form.get('HTR', 'D')
                
                # Get team stats
                team_stats = get_team_stats()
                
                # Make advanced prediction
                prediction_result = make_advanced_prediction(home_team, away_team, match_data, team_stats)
                
                return jsonify(prediction_result)
                
            except Exception as e:
                return jsonify({'success': False, 'error': f'Prediction error: {str(e)}'})
        
        # GET request - show form
        team_stats = get_team_stats()
        teams = team_stats.index.tolist() if not team_stats.empty else ['Arsenal', 'Chelsea', 'Liverpool', 'Man City', 'Man United', 'Tottenham']
        return render_template('predict.html', teams=teams)
    
    @app.route('/dashboard')
    @login_required
    def dashboard():
        # Get available plots
        plots_dir = 'outputs'
        available_plots = []
        
        if os.path.exists(plots_dir):
            plot_files = [f for f in os.listdir(plots_dir) if f.endswith('.png')]
            available_plots = [{'filename': f, 'title': f.replace('_', ' ').replace('.png', '').title()} 
                             for f in plot_files]
        
        team_stats = get_team_stats()
        
        return render_template('dashboard.html', 
                             plots=available_plots,
                             team_stats=team_stats.head(10).to_dict('records') if not team_stats.empty else [])
    
    @app.route('/team_analysis')
    @login_required
    def team_analysis():
        team_stats = get_team_stats()
        teams = team_stats.index.tolist() if not team_stats.empty else []
        
        team1 = request.args.get('team1')
        team2 = request.args.get('team2')
        
        comparison_data = None
        if team1 and team2 and team1 in teams and team2 in teams:
            comparison_data = {
                'team1': {
                    'name': team1,
                    'stats': team_stats.loc[team1].to_dict()
                },
                'team2': {
                    'name': team2,
                    'stats': team_stats.loc[team2].to_dict()
                }
            }
        
        return render_template('team_analysis.html', 
                             teams=teams,
                             comparison=comparison_data)
    
    @app.route('/chat', methods=['GET', 'POST'])
    @login_required
    def chat():
        if request.method == 'POST':
            try:
                user_message = request.json.get('message', '')
                
                if not user_message:
                    return safe_jsonify(create_error_response('No message provided', 'MISSING_MESSAGE'))
                
                # Get AI response
                ai_response = get_ai_response(user_message)
                
                return safe_jsonify({
                    'success': True,
                    'response': ai_response,
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as e:
                return safe_jsonify(create_error_response(f'Chat error: {str(e)}', 'CHAT_ERROR'))
        
        return render_template('chat.html')
    
    def get_ai_response(user_message):
        """Sports Match Outcome Chat Assistant"""
        
        try:
            # Load team stats for context
            team_stats = get_team_stats()
            
            # Sports Match Outcome Assistant context
            context = f"""
            You are an intelligent Sports Match Outcome Chat Assistant. Your purpose is to understand user questions about sports matches, teams, and outcomes — and respond accurately, conversationally, and confidently.
            
            Available EPL teams and their points: {team_stats['Points'].to_dict() if not team_stats.empty else 'No data available'}
            
            Always interpret the user's question carefully. If the question lacks clarity or data (e.g., no date, league, or team name), ask one short clarifying question before answering.
            
            Always give a structured, conversational answer that includes:
            - Key facts or stats
            - A short reasoning/explanation  
            - A final outcome or prediction (if requested)
            
            Keep responses concise, friendly, and professional. Stay neutral — never guess wildly or give biased opinions.
            """
            
            # Try OpenAI API
            if openai_api_key and openai_api_key != 'your-openai-api-key-here':
                try:
                    client = openai.OpenAI(api_key=openai_api_key)
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": context},
                            {"role": "user", "content": user_message}
                        ],
                        max_tokens=300,
                        temperature=0.7
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    print(f"OpenAI API error: {e}")
                    return get_sports_assistant_response(user_message, team_stats)
            else:
                # Sports assistant fallback
                return get_sports_assistant_response(user_message, team_stats)
                
        except Exception as e:
            return f"⚽ I'm having trouble processing your request right now. Please try asking about team stats, match predictions, or recent form."
    
    def get_sports_assistant_response(message, team_stats):
        """Sports Match Outcome Chat Assistant - Structured responses"""
        
        message_lower = message.lower()
        
        # Handle prediction requests
        if any(word in message_lower for word in ['who will win', 'predict', 'prediction', 'outcome', 'result']):
            if 'vs' in message_lower or ' v ' in message_lower or 'between' in message_lower:
                return handle_sports_prediction(message, team_stats)
            else:
                return "⚽ I can help with match predictions! Try asking 'Who will win between Arsenal and Chelsea?' or specify two teams for analysis."
        
        # Handle team stats requests
        elif any(word in message_lower for word in ['stats', 'statistics', 'performance', 'form']):
            return handle_team_stats(message, team_stats)
        
        # Handle specific team queries
        elif any(team in message_lower for team in ['arsenal', 'liverpool', 'chelsea', 'manchester', 'man city', 'tottenham']):
            return handle_team_info(message, team_stats)
        
        # Handle table/points queries
        elif any(word in message_lower for word in ['table', 'points', 'standings', 'position']):
            return handle_table_query(team_stats)
        
        # Handle form queries
        elif 'form' in message_lower:
            return handle_form_query(message, team_stats)
        
        # Default response
        else:
            return "⚽ I can help with match predictions, team stats, and recent form. Try asking:\n• 'Who will win between Team A and Team B?'\n• 'What are Arsenal's recent stats?'\n• 'Show me the current table'"
    
    def handle_sports_prediction(message, team_stats):
        """Handle sports match prediction requests"""
        try:
            # Extract team names
            message_lower = message.lower()
            teams = ['arsenal', 'chelsea', 'liverpool', 'man city', 'manchester city', 'man united', 
                    'manchester united', 'tottenham', 'newcastle', 'brighton', 'aston villa', 'west ham']
            
            found_teams = []
            for team in teams:
                if team in message_lower:
                    found_teams.append(team.replace('man city', 'Manchester City').replace('man united', 'Manchester United').title())
            
            if len(found_teams) >= 2:
                team1, team2 = found_teams[0], found_teams[1]
                
                # Get team stats if available
                team1_stats = team_stats.loc[team1] if not team_stats.empty and team1 in team_stats.index else None
                team2_stats = team_stats.loc[team2] if not team_stats.empty and team2 in team_stats.index else None
                
                response = f"⚽ **Match Prediction: {team1} vs {team2}**\n\n"
                
                if team1_stats is not None and team2_stats is not None:
                    # Compare stats
                    team1_points = int(team1_stats.get('Points', 0))
                    team2_points = int(team2_stats.get('Points', 0))
                    team1_wins = int(team1_stats.get('Wins', 0))
                    team2_wins = int(team2_stats.get('Wins', 0))
                    
                    response += f"**Recent Performance:**\n"
                    response += f"• {team1}: {team1_points} points, {team1_wins} wins\n"
                    response += f"• {team2}: {team2_points} points, {team2_wins} wins\n\n"
                    
                    # Simple prediction logic
                    if team1_points > team2_points:
                        winner = team1
                        confidence = min(65 + abs(team1_points - team2_points), 80)
                    elif team2_points > team1_points:
                        winner = team2
                        confidence = min(65 + abs(team2_points - team1_points), 80)
                    else:
                        winner = "Draw"
                        confidence = 45
                    
                    response += f"**Prediction:** {winner} ({confidence}% confidence)\n\n"
                    response += f"**Reasoning:** Based on current points and recent form analysis."
                else:
                    response += f"**Prediction:** Close match expected\n\n"
                    response += f"**Reasoning:** Both teams have competitive records. Use the AI Predictions page for detailed statistical analysis."
                
                return response
            else:
                return "⚽ Please specify two teams for prediction. Example: 'Who will win between Arsenal and Chelsea?'"
                
        except Exception as e:
            return "⚽ I had trouble processing that prediction. Please try asking about specific teams or use the AI Predictions page."
    
    def handle_team_stats(message, team_stats):
        """Handle team statistics requests"""
        message_lower = message.lower()
        
        for team_name in team_stats.index if not team_stats.empty else []:
            if team_name.lower() in message_lower:
                stats = team_stats.loc[team_name]
                response = f"⚽ **{team_name} Statistics:**\n\n"
                response += f"• Points: {int(stats.get('Points', 0))}\n"
                response += f"• Wins: {int(stats.get('Wins', 0))}\n"
                response += f"• Draws: {int(stats.get('Draws', 0))}\n"
                response += f"• Losses: {int(stats.get('Losses', 0))}\n"
                response += f"• Goals For: {int(stats.get('GoalsFor', 0))}\n"
                response += f"• Goals Against: {int(stats.get('GoalsAgainst', 0))}\n\n"
                
                # Form assessment
                win_rate = stats.get('Wins', 0) / max(stats.get('Wins', 0) + stats.get('Draws', 0) + stats.get('Losses', 0), 1) * 100
                if win_rate > 60:
                    form = "excellent"
                elif win_rate > 40:
                    form = "good"
                else:
                    form = "needs improvement"
                
                response += f"**Current Form:** {form.title()} ({win_rate:.1f}% win rate)"
                return response
        
        return "⚽ Please specify a team name for statistics. Available teams: " + ", ".join(team_stats.index.tolist()[:5]) if not team_stats.empty else "No team data available."
    
    def handle_team_info(message, team_stats):
        """Handle general team information requests"""
        message_lower = message.lower()
        
        if 'arsenal' in message_lower:
            if not team_stats.empty and 'Arsenal' in team_stats.index:
                stats = team_stats.loc['Arsenal']
                return f"⚽ **Arsenal:** Currently {int(stats.get('Points', 0))} points with {int(stats.get('Wins', 0))} wins. Known for their attacking style and Emirates Stadium home advantage."
            return "⚽ **Arsenal:** Historic North London club known for their attacking football and strong home record at Emirates Stadium."
        
        elif 'liverpool' in message_lower:
            if not team_stats.empty and 'Liverpool' in team_stats.index:
                stats = team_stats.loc['Liverpool']
                return f"⚽ **Liverpool:** Currently {int(stats.get('Points', 0))} points with {int(stats.get('Wins', 0))} wins. Famous for their high-intensity pressing game and Anfield atmosphere."
            return "⚽ **Liverpool:** Historic club with passionate fanbase, known for high-intensity pressing and strong European pedigree."
        
        elif 'chelsea' in message_lower:
            if not team_stats.empty and 'Chelsea' in team_stats.index:
                stats = team_stats.loc['Chelsea']
                return f"⚽ **Chelsea:** Currently {int(stats.get('Points', 0))} points with {int(stats.get('Wins', 0))} wins. Known for tactical flexibility and strong defensive organization."
            return "⚽ **Chelsea:** West London club known for tactical flexibility, strong defense, and rich history of success."
        
        return "⚽ I can provide information about Premier League teams. Which team would you like to know about?"
    
    def handle_table_query(team_stats):
        """Handle league table requests"""
        if team_stats.empty:
            return "⚽ I don't have current table data available."
        
        top_teams = team_stats.head(5)
        response = "⚽ **Current Top 5 Teams:**\n\n"
        for i, (team, stats) in enumerate(top_teams.iterrows(), 1):
            response += f"{i}. {team}: {int(stats['Points'])} points ({int(stats['Wins'])}W-{int(stats['Draws'])}D-{int(stats['Losses'])}L)\n"
        
        response += "\nWould you like detailed stats for any specific team?"
        return response
    
    def handle_form_query(message, team_stats):
        """Handle team form requests"""
        message_lower = message.lower()
        
        for team_name in team_stats.index if not team_stats.empty else []:
            if team_name.lower() in message_lower:
                stats = team_stats.loc[team_name]
                wins = int(stats.get('Wins', 0))
                total_games = wins + int(stats.get('Draws', 0)) + int(stats.get('Losses', 0))
                win_rate = (wins / max(total_games, 1)) * 100
                
                if win_rate > 60:
                    form_desc = "excellent"
                elif win_rate > 40:
                    form_desc = "good"
                else:
                    form_desc = "inconsistent"
                
                return f"⚽ **{team_name}'s Form:** {form_desc.title()} - {wins} wins in {total_games} matches ({win_rate:.1f}% win rate). Current points: {int(stats.get('Points', 0))}"
        
        return "⚽ Please specify a team name for form analysis."
    
    @app.route('/api/plot/<filename>')
    def serve_plot(filename):
        """Serve plot images"""
        from flask import send_from_directory
        return send_from_directory('../outputs', filename)
    
    def make_advanced_prediction(home_team, away_team, match_data, team_stats):
        """Make advanced prediction with all match statistics"""
        try:
            # Get team stats if available
            if not team_stats.empty and home_team in team_stats.index and away_team in team_stats.index:
                home_stats = team_stats.loc[home_team]
                away_stats = team_stats.loc[away_team]
                
                home_strength = home_stats['Points'] + (home_stats['GoalsFor'] - home_stats['GoalsAgainst']) * 0.1
                away_strength = away_stats['Points'] + (away_stats['GoalsFor'] - away_stats['GoalsAgainst']) * 0.1
            else:
                # Default values if teams not found
                home_strength = 60
                away_strength = 55
                home_stats = {'Points': 60, 'Wins': 15, 'Draws': 8, 'Losses': 15, 'GoalsFor': 45, 'GoalsAgainst': 40}
                away_stats = {'Points': 55, 'Wins': 14, 'Draws': 7, 'Losses': 17, 'GoalsFor': 42, 'GoalsAgainst': 45}
            
            # Factor in match statistics
            home_match_strength = (
                match_data['HS'] * 0.1 + match_data['HST'] * 0.2 + 
                match_data['HC'] * 0.05 + match_data['HTHG'] * 0.3
            )
            away_match_strength = (
                match_data['AS'] * 0.1 + match_data['AST'] * 0.2 + 
                match_data['AC'] * 0.05 + match_data['HTAG'] * 0.3
            )
            
            # Combine team and match strengths
            total_home_strength = home_strength + home_match_strength + 5  # Home advantage
            total_away_strength = away_strength + away_match_strength
            
            # Half-time influence
            ht_influence = 0
            if match_data['HTR'] == 'H':
                ht_influence = 10
            elif match_data['HTR'] == 'A':
                ht_influence = -10
            
            total_home_strength += ht_influence
            
            # Calculate probabilities
            strength_diff = total_home_strength - total_away_strength
            
            if strength_diff > 15:
                home_prob = 0.65
                draw_prob = 0.20
                away_prob = 0.15
                prediction = 'H'
            elif strength_diff < -10:
                home_prob = 0.20
                draw_prob = 0.25
                away_prob = 0.55
                prediction = 'A'
            elif abs(strength_diff) <= 5:
                home_prob = 0.35
                draw_prob = 0.35
                away_prob = 0.30
                prediction = 'D'
            elif strength_diff > 0:
                home_prob = 0.50
                draw_prob = 0.30
                away_prob = 0.20
                prediction = 'H'
            else:
                home_prob = 0.25
                draw_prob = 0.30
                away_prob = 0.45
                prediction = 'A'
            
            confidence = max(home_prob, draw_prob, away_prob) * 100
            
            # Predicted scores
            home_goals = max(0, round(match_data['HTHG'] + (home_match_strength / 10)))
            away_goals = max(0, round(match_data['HTAG'] + (away_match_strength / 10)))
            
            return {
                'success': True,
                'prediction': prediction,
                'confidence': round(confidence, 1),
                'probabilities': {
                    'H': round(home_prob, 3),
                    'D': round(draw_prob, 3),
                    'A': round(away_prob, 3)
                },
                'predicted_score': f"{home_goals}-{away_goals}",
                'home_team': home_team,
                'away_team': away_team,
                'team_stats': {
                    'home': home_stats if isinstance(home_stats, dict) else home_stats.to_dict(),
                    'away': away_stats if isinstance(away_stats, dict) else away_stats.to_dict()
                },
                'match_analysis': {
                    'home_shots': int(match_data['HS']),
                    'away_shots': int(match_data['AS']),
                    'home_shots_target': int(match_data['HST']),
                    'away_shots_target': int(match_data['AST']),
                    'halftime_score': f"{int(match_data['HTHG'])}-{int(match_data['HTAG'])}",
                    'halftime_result': match_data['HTR']
                }
            }
        
        except Exception as e:
            return {'success': False, 'error': f'Prediction failed: {str(e)}'}
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)