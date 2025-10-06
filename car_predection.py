import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv(r"C:\Users\chapa mahindra\Downloads\archive (1)\car details v4.csv")

print("Dataset Info:")
print(f"Total cars: {len(df)}")
print(f"Unique models: {df['Model'].nunique()}")

# Data preprocessing
def preprocess_data(df):
    df_processed = df.copy()
    
    # Handle missing values
    df_processed = df_processed.fillna(method='ffill').fillna(method='bfill')
    
    # Extract numerical values from engine, power, torque
    df_processed['Engine'] = df_processed['Engine'].astype(str).str.extract(r'(\d+\.?\d*)')[0].astype(float)
    df_processed['Max Power'] = df_processed['Max Power'].astype(str).str.extract(r'(\d+\.?\d*)')[0].astype(float)
    df_processed['Max Torque'] = df_processed['Max Torque'].astype(str).str.extract(r'(\d+\.?\d*)')[0].astype(float)
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = ['Make', 'Fuel Type', 'Transmission', 'Location', 'Color', 
                       'Owner', 'Seller Type', 'Drivetrain']
    
    for col in categorical_cols:
        df_processed[col] = le.fit_transform(df_processed[col].astype(str))
    
    return df_processed

df_processed = preprocess_data(df)

# Feature engineering for recommendation
def create_recommendation_features(df):
    """Create features that will be used for filtering recommendations"""
    features = [
        'Make', 'Year', 'Kilometer', 'Fuel Type', 'Transmission', 
        'Location', 'Color', 'Owner', 'Seller Type', 'Engine', 
        'Max Power', 'Max Torque', 'Drivetrain', 'Length', 
        'Width', 'Height', 'Seating Capacity', 'Fuel Tank Capacity', 'Price'
    ]
    return df[features]

X = create_recommendation_features(df_processed)

# Get top N most popular models for recommendation
def get_top_models(df, top_n=50):
    """Get top N most frequent models to reduce complexity"""
    model_counts = df['Model'].value_counts()
    top_models = model_counts.head(top_n).index
    return top_models

top_models = get_top_models(df, top_n=50)
df_filtered = df_processed[df_processed['Model'].isin(top_models)]

print(f"After filtering: {len(df_filtered)} cars, {df_filtered['Model'].nunique()} unique models")

# Prepare data for training
X_filtered = create_recommendation_features(df_filtered)
y_filtered = df_filtered['Model']

# Encode target
le_model = LabelEncoder()
y_encoded = le_model.fit_transform(y_filtered)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_filtered, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Handle missing values and scale
numeric_cols = X_train.select_dtypes(include=[np.number]).columns
si = SimpleImputer(strategy='median')
sc = StandardScaler()

X_train[numeric_cols] = si.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = si.transform(X_test[numeric_cols])

X_train[numeric_cols] = sc.fit_transform(X_train[numeric_cols])
X_test[numeric_cols] = sc.transform(X_test[numeric_cols])

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Recommendation function
def recommend_cars(user_preferences, model, le_model, df_original, top_k=10):
    """
    Recommend top K cars based on user preferences
    
    user_preferences: dict with keys like:
        {
            'Make': 'Toyota',
            'Year': 2020,
            'Price': 1500000,
            'Fuel Type': 'Petrol',
            'Transmission': 'Manual',
            'Kilometer': 15000,
            'Owner': 'First'
        }
    """
    # Create a template row with default values
    template = {
        'Make': 0, 'Year': 2020, 'Kilometer': 50000, 'Fuel Type': 0, 
        'Transmission': 0, 'Location': 0, 'Color': 0, 'Owner': 0, 
        'Seller Type': 0, 'Engine': 1500, 'Max Power': 100, 
        'Max Torque': 200, 'Drivetrain': 0, 'Length': 4000, 
        'Width': 1700, 'Height': 1500, 'Seating Capacity': 5, 
        'Fuel Tank Capacity': 40, 'Price': 1000000
    }
    
    # Update template with user preferences
    template.update(user_preferences)
    
    # Encode categorical variables if they're strings
    le = LabelEncoder()
    categorical_mapping = {
        'Make': dict(zip(df_original['Make'].unique(), range(len(df_original['Make'].unique())))),
        'Fuel Type': dict(zip(df_original['Fuel Type'].unique(), range(len(df_original['Fuel Type'].unique())))),
        'Transmission': dict(zip(df_original['Transmission'].unique(), range(len(df_original['Transmission'].unique())))),
        'Owner': dict(zip(df_original['Owner'].unique(), range(len(df_original['Owner'].unique()))))
    }
    
    for key, value in template.items():
        if key in categorical_mapping and isinstance(value, str):
            if value in categorical_mapping[key]:
                template[key] = categorical_mapping[key][value]
            else:
                template[key] = 0  # Default value if not found
    
    # Create input array
    input_features = np.array([list(template.values())])
    
    # Get probabilities for all classes
    probabilities = model.predict_proba(input_features)[0]
    
    # Get top K recommendations
    top_k_indices = np.argsort(probabilities)[-top_k:][::-1]
    top_k_models = le_model.inverse_transform(top_k_indices)
    top_k_probs = probabilities[top_k_indices]
    
    # Get details of recommended cars
    recommendations = []
    for model_name, prob in zip(top_k_models, top_k_probs):
        model_cars = df_original[df_original['Model'] == model_name]
        if len(model_cars) > 0:
            car_details = model_cars.iloc[0]
            recommendations.append({
                'Model': model_name,
                'Make': car_details['Make'],
                'Year': car_details['Year'],
                'Price': car_details['Price'],
                'Fuel Type': car_details['Fuel Type'],
                'Transmission': car_details['Transmission'],
                'Confidence': f"{prob:.2%}",
                'Kilometer': car_details['Kilometer']
            })
    
    return recommendations[:top_k]

# Example usage
if __name__ == "__main__":
    # Example user preferences
    user_prefs = {
        'Make': 'Toyota',
        'Year': 2020,
        'Price': 1500000,
        'Fuel Type': 'Petrol',
        'Transmission': 'Manual',
        'Kilometer': 15000,
        'Owner': 'First'
    }
    
    print("=== Car Recommendation System ===")
    print("User Preferences:")
    for key, value in user_prefs.items():
        print(f"{key}: {value}")
    
    print("\nTop 10 Recommended Cars:")
    recommendations = recommend_cars(user_prefs, model, le_model, df, top_k=10)
    
    for i, car in enumerate(recommendations, 1):
        print(f"\n{i}. {car['Make']} {car['Model']} ({car['Year']})")
        print(f"   Price: ₹{car['Price']:,.0f}")
        print(f"   Fuel: {car['Fuel Type']} | Transmission: {car['Transmission']}")
        print(f"   Kilometer: {car['Kilometer']} | Confidence: {car['Confidence']}")

# Alternative: Content-based filtering approach
def content_based_recommendation(df, user_preferences, top_k=10):
    """
    Simple content-based filtering without ML model
    """
    filtered_cars = df.copy()
    
    # Filter based on user preferences
    if 'Make' in user_preferences:
        filtered_cars = filtered_cars[filtered_cars['Make'] == user_preferences['Make']]
    if 'Fuel Type' in user_preferences:
        filtered_cars = filtered_cars[filtered_cars['Fuel Type'] == user_preferences['Fuel Type']]
    if 'Transmission' in user_preferences:
        filtered_cars = filtered_cars[filtered_cars['Transmission'] == user_preferences['Transmission']]
    
    # Price range filter (within 20% of desired price)
    if 'Price' in user_preferences:
        desired_price = user_preferences['Price']
        price_range = desired_price * 0.2
        filtered_cars = filtered_cars[
            (filtered_cars['Price'] >= desired_price - price_range) & 
            (filtered_cars['Price'] <= desired_price + price_range)
        ]
    
    # Year filter (within 2 years)
    if 'Year' in user_preferences:
        desired_year = user_preferences['Year']
        filtered_cars = filtered_cars[
            (filtered_cars['Year'] >= desired_year - 2) & 
            (filtered_cars['Year'] <= desired_year + 2)
        ]
    
    # Sort by relevance (you can customize this)
    filtered_cars = filtered_cars.sort_values('Year', ascending=False)
    
    return filtered_cars.head(top_k)

# Test both approaches
print("\n" + "="*50)
print("Content-Based Filtering Results:")
content_based_recs = content_based_recommendation(df, user_prefs)
for i, (_, car) in enumerate(content_based_recs.iterrows(), 1):
    print(f"{i}. {car['Make']} {car['Model']} ({car['Year']}) - ₹{car['Price']:,.0f}")
