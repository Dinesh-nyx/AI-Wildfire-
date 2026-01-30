"""
Flask Web Application for AI-Driven Wildfire Detection
Backend server with ML model integration and API endpoints
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.secret_key = 'wildfire_detection_secret_key_2024'  # Change in production

# Simple admin credentials (in production, use database with hashed passwords)
ADMIN_CREDENTIALS = {
    'username': 'admin',
    'password': 'wildfire2024'
}

# Load ML models
print("Loading ML models...")
try:
    global_model = joblib.load('wildfire_best_model.pkl')
    regional_model = joblib.load('regional_wildfire_model.pkl')
    scaler = joblib.load('wildfire_scaler.pkl')
    feature_names = joblib.load('wildfire_feature_names.pkl')
    print("✓ Models loaded successfully")
except Exception as e:
    print(f"Error loading models: {e}")
    global_model = None

# Risk assessment configuration
RISK_LEVELS = {
    'LOW': (0.0, 0.20),
    'MODERATE': (0.20, 0.40),
    'HIGH': (0.40, 0.60),
    'CRITICAL': (0.60, 0.80),
    'EXTREME': (0.80, 1.01)
}

RESPONSE_TIMES = {
    'EXTREME': 'IMMEDIATE (0-5 minutes)',
    'CRITICAL': 'URGENT (5-15 minutes)',
    'HIGH': '15-30 minutes',
    'MODERATE': '30-60 minutes',
    'LOW': 'Routine surveillance'
}

RECOMMENDATIONS = {
    'EXTREME': [
        'IMMEDIATE EVACUATION of all residents in affected areas',
        'DEPLOY all available firefighting resources',
        'ACTIVATE emergency response teams and incident command',
        'ESTABLISH emergency shelters and medical facilities',
        'REQUEST mutual aid from neighboring jurisdictions'
    ],
    'CRITICAL': [
        'ALERT firefighting teams to standby positions',
        'PREPARE evacuation routes and emergency procedures',
        'MONITOR weather conditions continuously',
        'NOTIFY emergency services and hospitals',
        'POSITION resources at strategic locations'
    ],
    'HIGH': [
        'INCREASE surveillance and monitoring frequency',
        'PREPARE firefighting equipment and personnel',
        'NOTIFY local authorities',
        'REVIEW evacuation plans',
        'ASSESS available resources'
    ],
    'MODERATE': [
        'Maintain enhanced monitoring',
        'Brief emergency personnel on situation',
        'Check equipment readiness'
    ],
    'LOW': [
        'Continue routine surveillance',
        'Maintain standard protocols'
    ]
}

def assess_risk_level(probability):
    """Determine risk level based on fire probability"""
    for level, (low, high) in RISK_LEVELS.items():
        if low <= probability < high:
            return level
    return 'EXTREME'

@app.route('/')
def index():
    """Main dashboard page - NEW CLEAN MINIMALIST DESIGN"""
    return render_template('dashboard_new.html')

@app.route('/api/assessments')
def get_assessments():
    """Get existing risk assessments from CSV"""
    try:
        df = pd.read_csv('risk_assessment_results.csv')
        assessments = df.to_dict('records')
        return jsonify(assessments)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make wildfire prediction for custom location"""
    try:
        data = request.json
        
        # Validate input
        required_fields = ['NBR', 'NDMI', 'NDVI', 'aspect', 'elevation', 
                          'landcover', 'rain', 'slope', 'temperature', 
                          'latitude', 'longitude']
        
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Create DataFrame
        input_df = pd.DataFrame([{
            'NBR': float(data['NBR']),
            'NDMI': float(data['NDMI']),
            'NDVI': float(data['NDVI']),
            'aspect': float(data['aspect']),
            'elevation': int(data['elevation']),
            'landcover': int(data['landcover']),
            'rain': float(data['rain']),
            'slope': float(data['slope']),
            'temperature': float(data['temperature']),
            'latitude': float(data['latitude']),
            'longitude': float(data['longitude'])
        }])
        
        # Ensure correct feature order
        input_df = input_df[feature_names]
        
        # Scale features
        X_scaled = scaler.transform(input_df)
        
        # Make prediction
        prediction = global_model.predict(X_scaled)[0]
        probability = global_model.predict_proba(X_scaled)[0][1]
        
        # Assess risk
        risk_level = assess_risk_level(probability)
        
        result = {
            'prediction': 'FIRE' if prediction == 1 else 'NO FIRE',
            'fire_probability': float(probability),
            'risk_level': risk_level,
            'response_time': RESPONSE_TIMES[risk_level],
            'recommendations': RECOMMENDATIONS[risk_level],
            'confidence': float(max(global_model.predict_proba(X_scaled)[0])),
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/examples')
def get_examples():
    """Get example input data"""
    examples = {
        'high': {
            'latitude': -37.10,
            'longitude': 149.80,
            'elevation': 350,
            'aspect': 140,
            'slope': 14.5,
            'landcover': 10,
            'temperature': 35.0,
            'rain': 10.5,
            'NDVI': 0.25,
            'NBR': 0.20,
            'NDMI': 0.15
        },
        'moderate': {
            'latitude': -37.12,
            'longitude': 149.75,
            'elevation': 340,
            'aspect': 150,
            'slope': 12.0,
            'landcover': 12,
            'temperature': 22.0,
            'rain': 35.0,
            'NDVI': 0.55,
            'NBR': 0.45,
            'NDMI': 0.35
        },
        'low': {
            'latitude': -37.08,
            'longitude': 149.85,
            'elevation': 330,
            'aspect': 130,
            'slope': 10.0,
            'landcover': 8,
            'temperature': 18.0,
            'rain': 50.0,
            'NDVI': 0.75,
            'NBR': 0.65,
            'NDMI': 0.55
        }
    }
    return jsonify(examples)

@app.route('/api/stats')
def get_stats():
    """Get dashboard statistics"""
    try:
        df = pd.read_csv('risk_assessment_results.csv')
        
        stats = {
            'total_locations': len(df),
            'active_alerts': len(df[df['risk_level'].isin(['HIGH', 'CRITICAL', 'EXTREME'])]),
            'extreme_risk': len(df[df['risk_level'] == 'EXTREME']),
            'high_risk': len(df[df['risk_level'].isin(['HIGH', 'CRITICAL', 'EXTREME'])])
        }
        
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    """Chatbot endpoint for wildfire queries"""
    try:
        data = request.json
        message = data.get('message', '').lower()
        
        # Simple rule-based responses
        responses = {
            'hello': 'Hello! I\'m your Wildfire Detection Assistant. How can I help you today?',
            'hi': 'Hi there! Ask me about wildfire risks, predictions, or safety measures.',
            'help': 'I can help you with:\n• Fire risk predictions\n• Safety recommendations\n• Current alerts\n• Regional statistics\n\nWhat would you like to know?',
            'risk': 'Current risk levels vary by region. Check the map for detailed information. Would you like statistics for a specific area?',
            'alert': f'There are currently {get_stats().json["active_alerts"]} active alerts. Check the dashboard for details.',
            'safety': 'Safety tips:\n• Stay informed about local conditions\n• Have an evacuation plan\n• Keep emergency supplies ready\n• Follow official warnings\n• Never ignore evacuation orders',
            'predict': 'I can help predict fire risk! Please provide location details or use the prediction form on the dashboard.',
            'australia': 'The system monitors wildfire risks across Victoria, Australia, including East Gippsland, Alpine National Park, Snowy River, and High Country regions.',
        }
        
        # Find matching response
        response_text = 'I\'m here to help with wildfire detection and safety. Try asking about "risk", "alerts", "safety", or "predictions".'
        for keyword, response in responses.items():
            if keyword in message:
                response_text = response
                break
        
        return jsonify({
            'response': response_text,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/login', methods=['POST'])
def admin_login():
    """Admin login endpoint"""
    try:
        data = request.json
        username = data.get('username')
        password = data.get('password')
        
        if username == ADMIN_CREDENTIALS['username'] and password == ADMIN_CREDENTIALS['password']:
            session['admin_logged_in'] = True
            session['admin_username'] = username
            return jsonify({'success': True, 'message': 'Login successful'})
        else:
            return jsonify({'success': False, 'message': 'Invalid credentials'}), 401
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/logout', methods=['POST'])
def admin_logout():
    """Admin logout endpoint"""
    session.pop('admin_logged_in', None)
    session.pop('admin_username', None)
    return jsonify({'success': True, 'message': 'Logged out successfully'})

@app.route('/api/admin/check')
def admin_check():
    """Check if admin is logged in"""
    return jsonify({
        'logged_in': session.get('admin_logged_in', False),
        'username': session.get('admin_username', '')
    })

@app.route('/api/admin/analytics')
def admin_analytics():
    """Get detailed analytics for admin panel"""
    if not session.get('admin_logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        df = pd.read_csv('risk_assessment_results.csv')
        
        analytics = {
            'total_locations': len(df),
            'risk_distribution': {
                'EXTREME': len(df[df['risk_level'] == 'EXTREME']),
                'CRITICAL': len(df[df['risk_level'] == 'CRITICAL']),
                'HIGH': len(df[df['risk_level'] == 'HIGH']),
                'MODERATE': len(df[df['risk_level'] == 'MODERATE']),
                'LOW': len(df[df['risk_level'] == 'LOW'])
            },
            'average_temperature': float(df['temperature'].mean()),
            'average_rainfall': float(df['rain'].mean()),
            'average_ndvi': float(df['NDVI'].mean()),
            'highest_risk_locations': df.nlargest(5, 'fire_probability')[
                ['sample_id', 'latitude', 'longitude', 'fire_probability', 'risk_level']
            ].to_dict('records'),
            'environmental_summary': {
                'temp_range': [float(df['temperature'].min()), float(df['temperature'].max())],
                'rain_range': [float(df['rain'].min()), float(df['rain'].max())],
                'elevation_range': [int(df['elevation'].min()), int(df['elevation'].max())]
            }
        }
        
        return jsonify(analytics)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/admin/send-alert', methods=['POST'])
def send_alert():
    """Send SMS alert to region (simulated)"""
    if not session.get('admin_logged_in'):
        return jsonify({'error': 'Unauthorized'}), 401
    
    try:
        data = request.json
        region = data.get('region')
        level = data.get('level')
        message = data.get('message')
        
        # Validate inputs
        if not all([region, level, message]):
            return jsonify({'success': False, 'message': 'Missing required fields'}), 400
        
        # In production, integrate with SMS service (Twilio, AWS SNS, etc.)
        # For now, simulate sending and log the alert
        
        # Calculate estimated recipients based on region
        df = pd.read_csv('risk_assessment_results.csv')
        
        if region == 'all':
            recipients_count = len(df) * 100  # Assume 100 people per location
        else:
            # Filter by region
            filtered = df[df['sample_id'].apply(lambda x: str(int(x) % 4) == region if region in ['0', '10', '73', '33'] else True)]
            recipients_count = len(filtered) * 100
        
        # Log the alert (in production, save to database)
        alert_log = {
            'timestamp': datetime.now().isoformat(),
            'admin': session.get('admin_username'),
            'region': region,
            'level': level,
            'message': message,
            'recipients': recipients_count
        }
        
        print(f"\n{'='*80}")
        print(f"🚨 SMS ALERT SENT")
        print(f"{'='*80}")
        print(f"Admin: {alert_log['admin']}")
        print(f"Region: {region}")
        print(f"Level: {level.upper()}")
        print(f"Recipients: ~{recipients_count}")
        print(f"Message:\n{message}")
        print(f"{'='*80}\n")
        
        # In production, call SMS API here:
        # sms_service.send_bulk(recipients, message)
        
        return jsonify({
            'success': True,
            'message': 'Alert sent successfully',
            'recipients_count': recipients_count,
            'timestamp': alert_log['timestamp']
        })
        
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

@app.route('/admin')
def admin_panel():
    """Admin panel page - NEW CLEAN MINIMALIST DESIGN"""
    return render_template('admin_new.html')

if __name__ == '__main__':
    print("=" * 80)
    print("🔥 WILDFIRE DETECTION FLASK SERVER")
    print("=" * 80)
    print("\n✅ Server starting...")
    print("📊 Dashboard: http://localhost:5000")
    print("🔥 API Base: http://localhost:5000/api")
    print("👤 Admin Panel: http://localhost:5000/admin")
    print("\nAvailable endpoints:")
    print("  GET  /                - Dashboard interface")
    print("  GET  /admin           - Admin panel")
    print("  GET  /api/assessments - Get existing assessments")
    print("  POST /api/predict     - Make custom prediction")
    print("  POST /api/chatbot     - Chatbot queries")
    print("  POST /api/admin/login - Admin login")
    print("  GET  /api/admin/analytics - Detailed analytics")
    print("  GET  /api/examples    - Get example data")
    print("  GET  /api/stats       - Get statistics")
    print("\n📝 Admin Credentials:")
    print("  Username: admin")
    print("  Password: wildfire2024")
    print("\n" + "=" * 80)
    
    app.run(debug=True, host='0.0.0.0', port=5000)

