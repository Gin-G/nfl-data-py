from flask import Flask, render_template, request
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json

app = Flask(__name__)

# Load the data
df = pd.read_csv('../data/nfl_dataset.csv')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/team_snap_counts')
def team_snap_counts():
    # Group by team and calculate average snap counts
    team_snaps = df.groupby('recent_team').agg({
        'offensive_snaps': 'mean',
        'defensive_snaps': 'mean',
        'special_teams_snaps': 'mean'
    }).reset_index()
    
    return render_template('team_snap_counts.html', teams=team_snaps.to_dict('records'))

@app.route('/player/<player_id>')
def player_page(player_id):
    player_data = df[df['player_id'] == player_id].sort_values('week')
    
    # Create a bar chart of fantasy points by week
    fig = px.bar(player_data, x='week', y='fanduel_fantasy_points', title='Fantasy Points by Week')
    chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('player.html', player=player_data.iloc[0], chart_json=chart_json)

@app.route('/position/<position>')
def position_page(position):
    position_data = df[df['position'] == position].groupby('player_id').agg({
        'fanduel_fantasy_points': 'mean',
        'offensive_snap_pct': 'mean',
        'defensive_snap_pct': 'mean',
        'special_teams_snap_pct': 'mean',
        'player_name': 'first'
    }).sort_values('fanduel_fantasy_points', ascending=False).head(20)
    
    return render_template('position.html', position=position, players=position_data.to_dict('records'))

if __name__ == '__main__':
    app.run(debug=True)