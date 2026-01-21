"""
Test script to verify the model is working correctly.
Run this from the Django project root: python test_model.py
"""
import os
import sys
import django

# Setup Django
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'EduSight.settings')
django.setup()

from core.ml_models.loader import ModelLoader
import datetime

def test_model_predictions():
    """Test the model with different inputs to verify it's working."""
    print("=" * 60)
    print("Testing Model Predictions")
    print("=" * 60)
    
    # Verify model loads
    is_valid, message = ModelLoader.verify_model('public_private_knn')
    print(f"\nModel Verification: {message}")
    if not is_valid:
        print("ERROR: Model verification failed!")
        return
    
    # Test case 1: Small school, Paris
    print("\n" + "-" * 60)
    print("Test Case 1: Small school in Paris")
    print("-" * 60)
    test_input_1 = {
        'nb_eleves': 50,
        'code_postal': '75001',
        'region': 'Île-de-France',
        'date': datetime.date(2024, 1, 15),
        'code_section': 'Unknown',
        'code_voie': 'Unknown',
        'code_type_etab': 'Unknown',
        'code_service': 'Unknown'
    }
    result_1 = ModelLoader.predict('public_private_knn', test_input_1)
    print(f"Input: {test_input_1}")
    print(f"Prediction: {result_1}")
    
    # Test case 2: Large school, different region
    print("\n" + "-" * 60)
    print("Test Case 2: Large school in different region")
    print("-" * 60)
    test_input_2 = {
        'nb_eleves': 500,
        'code_postal': '69001',
        'region': 'Auvergne-Rhône-Alpes',
        'date': datetime.date(2024, 6, 20),
        'code_section': 'Unknown',
        'code_voie': 'Unknown',
        'code_type_etab': 'Unknown',
        'code_service': 'Unknown'
    }
    result_2 = ModelLoader.predict('public_private_knn', test_input_2)
    print(f"Input: {test_input_2}")
    print(f"Prediction: {result_2}")
    
    # Test case 3: Medium school, different date
    print("\n" + "-" * 60)
    print("Test Case 3: Medium school, different date")
    print("-" * 60)
    test_input_3 = {
        'nb_eleves': 200,
        'code_postal': '13001',
        'region': "Provence-Alpes-Côte d'Azur",
        'date': datetime.date(2024, 9, 1),
        'code_section': 'Unknown',
        'code_voie': 'Unknown',
        'code_type_etab': 'Unknown',
        'code_service': 'Unknown'
    }
    result_3 = ModelLoader.predict('public_private_knn', test_input_3)
    print(f"Input: {test_input_3}")
    print(f"Prediction: {result_3}")
    
    # Test case 4: Very large school (more likely to be public)
    print("\n" + "-" * 60)
    print("Test Case 4: Very large school (likely public)")
    print("-" * 60)
    test_input_4 = {
        'nb_eleves': 1000,
        'code_postal': '33000',
        'region': 'Nouvelle-Aquitaine',
        'date': datetime.date(2024, 3, 10),
        'code_section': 'Unknown',
        'code_voie': 'Unknown',
        'code_type_etab': 'Unknown',
        'code_service': 'Unknown'
    }
    result_4 = ModelLoader.predict('public_private_knn', test_input_4)
    print(f"Input: {test_input_4}")
    print(f"Prediction: {result_4}")
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Test 1 result: {result_1}")
    print(f"Test 2 result: {result_2}")
    print(f"Test 3 result: {result_3}")
    print(f"Test 4 result: {result_4}")
    
    # Check if results are different (proves model is working)
    results = [result_1, result_2, result_3, result_4]
    unique_results = set(results)
    
    if len(unique_results) > 1:
        print(f"\n[SUCCESS] Model is working correctly! Different inputs produce different predictions.")
        print(f"  Found {len(unique_results)} unique prediction(s): {unique_results}")
    elif len(unique_results) == 1:
        print(f"\n[INFO] All test cases produced the same prediction: {unique_results}")
        print("  This could mean:")
        print("  - The model is making consistent predictions based on the input features")
        print("  - These particular inputs may all be more likely to be classified as the same class")
        print("  - The model is working correctly, but try with more varied inputs to see different predictions")
        print("\n  To verify the model can predict both classes, try inputs that are more likely")
        print("  to be public schools (e.g., larger schools, different regions, etc.)")
    else:
        print("\n[ERROR] No valid predictions returned!")

if __name__ == '__main__':
    test_model_predictions()

