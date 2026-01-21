import os
import joblib
import pandas as pd
import numpy as np
from django.conf import settings
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# Monkey-patch SimpleImputer to add _fill_dtype if missing
# This fixes compatibility issues when models trained with older scikit-learn versions
# are loaded with newer versions that expect _fill_dtype attribute

def _ensure_fill_dtype(imputer):
    """Helper to ensure _fill_dtype exists on a SimpleImputer."""
    # Use object.__getattribute__ to check if _fill_dtype exists without triggering our patched __getattribute__
    try:
        object.__getattribute__(imputer, '_fill_dtype')
        # It exists, nothing to do
        return
    except AttributeError:
        # It doesn't exist, create it
        # Check statistics_ using object.__getattribute__ to avoid recursion
        try:
            stats = object.__getattribute__(imputer, 'statistics_')
            if stats is not None:
                try:
                    imputer.__dict__['_fill_dtype'] = np.asarray(stats).dtype
                except:
                    imputer.__dict__['_fill_dtype'] = np.float64
            else:
                imputer.__dict__['_fill_dtype'] = np.float64
        except AttributeError:
            # statistics_ doesn't exist either, just set default
            imputer.__dict__['_fill_dtype'] = np.float64

# Store original methods
_original_transform = SimpleImputer.transform
_original_fit_transform = SimpleImputer.fit_transform
_original_fit = SimpleImputer.fit

def _patched_transform(self, X):
    """Patched transform that ensures _fill_dtype exists."""
    _ensure_fill_dtype(self)
    return _original_transform(self, X)

def _patched_fit_transform(self, X, y=None):
    """Patched fit_transform that ensures _fill_dtype exists."""
    result = _original_fit_transform(self, X, y)
    _ensure_fill_dtype(self)
    return result

def _patched_fit(self, X, y=None):
    """Patched fit that ensures _fill_dtype exists."""
    result = _original_fit(self, X, y)
    _ensure_fill_dtype(self)
    return result

# Apply monkey-patches
SimpleImputer.transform = _patched_transform
SimpleImputer.fit_transform = _patched_fit_transform
SimpleImputer.fit = _patched_fit

# Add a property descriptor to handle direct _fill_dtype access
# This creates _fill_dtype on-the-fly if accessed but missing
_original_getattribute = SimpleImputer.__getattribute__

def _patched_getattribute(self, name):
    """Patched __getattribute__ to ensure _fill_dtype exists when accessed."""
    if name == '_fill_dtype':
        # Check if it exists using object.__getattribute__ to avoid recursion
        try:
            return object.__getattribute__(self, '_fill_dtype')
        except AttributeError:
            # It doesn't exist, create it
            _ensure_fill_dtype(self)
            return object.__getattribute__(self, '_fill_dtype')
    # For all other attributes, use the original __getattribute__
    return _original_getattribute(self, name)

SimpleImputer.__getattribute__ = _patched_getattribute

def _patch_simple_imputer(obj, visited=None):
    """
    Recursively patches SimpleImputer objects to add missing _fill_dtype attribute.
    This fixes compatibility issues when models trained with older scikit-learn versions
    are loaded with newer versions.
    """
    if visited is None:
        visited = set()
    
    # Avoid infinite recursion
    obj_id = id(obj)
    if obj_id in visited:
        return
    visited.add(obj_id)
    
    # Patch SimpleImputer directly
    if isinstance(obj, SimpleImputer):
        if not hasattr(obj, '_fill_dtype'):
            # Infer the fill dtype from the statistics_ attribute
            if hasattr(obj, 'statistics_') and obj.statistics_ is not None:
                try:
                    obj._fill_dtype = np.asarray(obj.statistics_).dtype
                except:
                    obj._fill_dtype = np.float64
            else:
                obj._fill_dtype = np.float64
        return
    
    # Handle Pipeline
    if isinstance(obj, Pipeline):
        for name, step in obj.steps:
            _patch_simple_imputer(step, visited)
        return
    
    # Handle ColumnTransformer
    if isinstance(obj, ColumnTransformer):
        for name, transformer, columns in obj.transformers:
            if transformer != 'drop':
                _patch_simple_imputer(transformer, visited)
        return
    
    # Recursively check all attributes
    if hasattr(obj, '__dict__'):
        for attr_name, attr_value in obj.__dict__.items():
            # Skip private attributes and methods
            if not attr_name.startswith('__'):
                try:
                    # Check if it's a SimpleImputer, Pipeline, or ColumnTransformer
                    if isinstance(attr_value, (SimpleImputer, Pipeline, ColumnTransformer)):
                        _patch_simple_imputer(attr_value, visited)
                    # Also check if it's a list/tuple that might contain transformers
                    elif isinstance(attr_value, (list, tuple)):
                        for item in attr_value:
                            if isinstance(item, (SimpleImputer, Pipeline, ColumnTransformer)):
                                _patch_simple_imputer(item, visited)
                            elif isinstance(item, tuple) and len(item) >= 2:
                                # Handle (name, transformer) tuples
                                if isinstance(item[1], (SimpleImputer, Pipeline, ColumnTransformer)):
                                    _patch_simple_imputer(item[1], visited)
                except (AttributeError, TypeError, ValueError):
                    pass
    
    # Also check dir() for any additional attributes we might have missed
    try:
        for attr_name in dir(obj):
            if not attr_name.startswith('_'):
                try:
                    attr_value = getattr(obj, attr_name)
                    if isinstance(attr_value, (SimpleImputer, Pipeline, ColumnTransformer)):
                        _patch_simple_imputer(attr_value, visited)
                except (AttributeError, TypeError, ValueError):
                    pass
    except:
        pass

class ModelLoader:
    _models = {}
    
    # Define your model filenames here. 
    # Key = Model Name (safe for URLs/Forms), Value = Filename.pkl
    MODEL_FILES = {
        'public_private_knn': 'knn_model_public_prive.pkl',
        'establishment_type_svm': 'ecole_college_lycee_complete_model_1.pkl',
        'taux_reussite_xgb': 'xgboost_tuned_taux_reussite_model_final.pkl',
        'taux_mentions_xgb': 'xgboost_taux_mentions_model.pkl',  # Update with actual filename
        'student_number_classifier': 'student_number_classifier_complete_lastonebro.pkl',
        'clustering': 'clustering_model_complete.pkl',
    }

    @classmethod
    def clear_cache(cls, model_key=None):
        """Clear the model cache. If model_key is None, clears all cached models."""
        if model_key is None:
            cls._models.clear()
        elif model_key in cls._models:
            del cls._models[model_key]
    
    @classmethod
    def verify_model(cls, model_key):
        """Verify that the model is loaded and working correctly."""
        model = cls.get_model(model_key)
        if model is None:
            return False, "Model not loaded"
        
        # Check if model has predict method
        if not hasattr(model, 'predict'):
            return False, f"Model does not have predict method. Type: {type(model)}"
        
        # Try to get model info
        model_type = type(model).__name__
        return True, f"Model loaded successfully. Type: {model_type}"
    
    @classmethod
    def get_model(cls, model_key):
        """Lazy loads the model only when requested."""
        if model_key not in cls.MODEL_FILES:
            raise ValueError(f"Model key '{model_key}' not found.")
            
        if model_key not in cls._models:
            file_name = cls.MODEL_FILES[model_key]
            # Assumes models are stored in core/ml_models/
            model_path = os.path.join(settings.BASE_DIR, 'core', 'ml_models', file_name)
            
            if not os.path.exists(model_path):
                 # Return None or raise cleaner error so UI can handle missing files gracefully
                 print(f"WARNING: Model file not found at {model_path}")
                 return None
                 
            print(f"Loading {model_key} from {model_path}...")
            loaded_data = joblib.load(model_path)
            
            # Handle different model formats
            # For establishment_type_svm, it's a dict with model and transformers
            # For taux_reussite_xgb, it might be a dict with model and scaler, or just the model
            # For public_private_knn, it's just the model/pipeline
            if isinstance(loaded_data, dict) and 'model' in loaded_data:
                # This is a dict format (establishment_type_svm or taux_reussite_xgb)
                model = loaded_data
                # Patch the pipeline/model inside the dict
                if 'model' in model:
                    model_obj = model['model']
                    if hasattr(model_obj, 'named_steps') or hasattr(model_obj, 'predict'):
                        _patch_simple_imputer(model_obj)
            else:
                # This is a direct model/pipeline (public_private_knn format or taux_reussite_xgb)
                model = loaded_data
                if hasattr(model, 'named_steps') or hasattr(model, 'predict'):
                    _patch_simple_imputer(model)
            
            cls._models[model_key] = model
        else:
            # Even if cached, ensure it's patched (in case patches were added after caching)
            cached_model = cls._models[model_key]
            if isinstance(cached_model, dict) and 'model' in cached_model:
                _patch_simple_imputer(cached_model['model'])
            else:
                _patch_simple_imputer(cached_model)
            
        return cls._models[model_key]

    @classmethod
    def predict(cls, model_key, input_data):
        """
        Runs prediction.
        input_data: dict (from form.cleaned_data)
        """
        model = cls.get_model(model_key)
        if model is None:
            return "Model file missing or failed to load."

        try:
            # Special handling for Taux de Réussite XGBoost Regression model
            if model_key == 'taux_reussite_xgb':
                # Load model (should contain model and possibly scaler)
                model_data = model
                if model_data is None:
                    return "Model not loaded correctly."
                
                # Handle different model formats
                if isinstance(model_data, dict):
                    xgb_model = model_data.get('model') or model_data.get('xgb_model') or model_data.get('pipeline')
                    scaler = model_data.get('scaler') or model_data.get('standard_scaler') or model_data.get('StandardScaler')
                else:
                    # Assume it's just the model (could be a pipeline with scaling built-in)
                    xgb_model = model_data
                    scaler = None
                
                if xgb_model is None:
                    return "XGBoost model not found."
                
                # Check if the model is a pipeline that includes scaling
                if hasattr(xgb_model, 'named_steps'):
                    _patch_simple_imputer(xgb_model)
                    # Check if pipeline has a scaler step
                    if scaler is None:
                        from sklearn.preprocessing import StandardScaler
                        for step_name, step_obj in xgb_model.named_steps.items():
                            if 'scale' in step_name.lower() or isinstance(step_obj, StandardScaler):
                                # Pipeline already includes scaling, no need for separate scaler
                                scaler = None  # Pipeline will handle scaling
                                break
                
                # Extract date for temporal features
                date_obj = input_data['date']
                
                # Handle Code_Postal - convert to numeric
                try:
                    code_postal_num = float(input_data['code_postal'])
                except:
                    code_postal_num = 75001.0  # Default to Paris
                
                # Get input values
                statut = input_data['statut_etablissement']
                region = input_data['region']
                code_type = input_data.get('code_type_etab', 'Missing')
                code_section = input_data.get('code_section', 'Missing')
                code_voie = input_data.get('code_voie', 'Missing')
                code_service = input_data.get('code_service', 'Missing')
                
                # Create DataFrame with features in the EXACT order expected by the model
                # Based on the error message, the model was trained with these features:
                feature_data = {
                    # Numerical features
                    'IPS': [input_data['ips']],
                    'Taux_Mentions': [input_data['taux_mentions']],
                    'Nb_eleves': [input_data['nb_eleves']],
                    'Code_Postal': [code_postal_num],
                    # Statut_Etablissement (one-hot) - order matters!
                    'Statut_Etablissement_Privé': [1 if statut == 'Privé' else 0],
                    'Statut_Etablissement_Public': [1 if statut == 'Public' else 0],
                    # Regions (one-hot) - ALL regions that model was trained with
                    'Region_Auvergne-Rhône-Alpes': [1 if region == 'Auvergne-Rhône-Alpes' else 0],
                    'Region_Bourgogne-Franche-Comté': [1 if region == 'Bourgogne-Franche-Comté' else 0],
                    'Region_Bretagne': [1 if region == 'Bretagne' else 0],
                    'Region_Centre-Val de Loire': [1 if region == 'Centre-Val de Loire' else 0],
                    'Region_Corse': [1 if region == 'Corse' else 0],
                    'Region_Grand Est': [1 if region == 'Grand Est' else 0],
                    'Region_Guadeloupe': [1 if region == 'Guadeloupe' else 0],
                    'Region_Guyane': [1 if region == 'Guyane' else 0],
                    'Region_Hauts-de-France': [1 if region == 'Hauts-de-France' else 0],
                    'Region_Ile-de-France': [1 if region == 'Ile-de-France' else 0],
                    'Region_La Réunion': [1 if region == 'La Réunion' else 0],
                    'Region_Martinique': [1 if region == 'Martinique' else 0],
                    'Region_Mayotte': [1 if region == 'Mayotte' else 0],
                    'Region_Normandie': [1 if region == 'Normandie' else 0],
                    'Region_Nouvelle-Aquitaine': [1 if region == 'Nouvelle-Aquitaine' else 0],
                    'Region_Occitanie': [1 if region == 'Occitanie' else 0],
                    'Region_Pays de la Loire': [1 if region == 'Pays de la Loire' else 0],
                    "Region_Provence-Alpes-Côte d'Azur": [1 if region == "Provence-Alpes-Côte d'Azur" else 0],
                    'Region_TOM et Collectivités territoriales': [1 if region == 'TOM et Collectivités territoriales' else 0],
                    # Code_TypeEtab (one-hot) - INCLUDING Missing
                    'Code_TypeEtab_C': [1 if code_type == 'C' else 0],
                    'Code_TypeEtab_E': [1 if code_type == 'E' else 0],
                    'Code_TypeEtab_L': [1 if code_type == 'L' else 0],
                    'Code_TypeEtab_Missing': [1 if code_type == 'Missing' else 0],
                    # Temporal features
                    'day_of_week': [date_obj.weekday()],  # 0=Monday, 6=Sunday
                    'day_of_year': [date_obj.timetuple().tm_yday],
                    'quarter': [(date_obj.month - 1) // 3 + 1],
                    # Code_Section (one-hot) - ALL sections from training
                    'Code_Section_ART': [1 if code_section == 'ART' else 0],
                    'Code_Section_CIN': [1 if code_section == 'CIN' else 0],
                    'Code_Section_EUR': [1 if code_section == 'EUR' else 0],
                    'Code_Section_INT': [1 if code_section == 'INT' else 0],
                    'Code_Section_Missing': [1 if code_section == 'Missing' else 0],
                    'Code_Section_SPO': [1 if code_section == 'SPO' else 0],
                    'Code_Section_THE': [1 if code_section == 'THE' else 0],
                    # Code_Voie (one-hot) - P, T (NOT GT, PRO)
                    'Code_Voie_G': [1 if code_voie == 'G' else 0],
                    'Code_Voie_Missing': [1 if code_voie == 'Missing' else 0],
                    'Code_Voie_P': [1 if code_voie == 'P' else 0],
                    'Code_Voie_T': [1 if code_voie == 'T' else 0],
                    # Code_Service (one-hot) - ALL services from training
                    'Code_Service_APP': [1 if code_service == 'APP' else 0],
                    'Code_Service_GRE': [1 if code_service == 'GRE' else 0],
                    'Code_Service_HEB': [1 if code_service == 'HEB' else 0],
                    'Code_Service_Missing': [1 if code_service == 'Missing' else 0],
                    'Code_Service_PBC': [1 if code_service == 'PBC' else 0],
                    'Code_Service_RES': [1 if code_service == 'RES' else 0],
                    'Code_Service_SEG': [1 if code_service == 'SEG' else 0],
                    'Code_Service_ULI': [1 if code_service == 'ULI' else 0],
                }
                
                # Create DataFrame with exact feature order
                # Define the exact column order expected by the model (from training)
                expected_columns = [
                    'IPS', 'Taux_Mentions', 'Nb_eleves', 'Code_Postal',
                    'Statut_Etablissement_Privé', 'Statut_Etablissement_Public',
                    'Region_Auvergne-Rhône-Alpes', 'Region_Bourgogne-Franche-Comté',
                    'Region_Bretagne', 'Region_Centre-Val de Loire', 'Region_Corse',
                    'Region_Grand Est', 'Region_Guadeloupe', 'Region_Guyane',
                    'Region_Hauts-de-France', 'Region_Ile-de-France', 'Region_La Réunion',
                    'Region_Martinique', 'Region_Mayotte', 'Region_Normandie',
                    'Region_Nouvelle-Aquitaine', 'Region_Occitanie', 'Region_Pays de la Loire',
                    "Region_Provence-Alpes-Côte d'Azur", 'Region_TOM et Collectivités territoriales',
                    'Code_TypeEtab_C', 'Code_TypeEtab_E', 'Code_TypeEtab_L', 'Code_TypeEtab_Missing',
                    'day_of_week', 'day_of_year', 'quarter',
                    'Code_Section_ART', 'Code_Section_CIN', 'Code_Section_EUR',
                    'Code_Section_INT', 'Code_Section_Missing', 'Code_Section_SPO', 'Code_Section_THE',
                    'Code_Voie_G', 'Code_Voie_Missing', 'Code_Voie_P', 'Code_Voie_T',
                    'Code_Service_APP', 'Code_Service_GRE', 'Code_Service_HEB',
                    'Code_Service_Missing', 'Code_Service_PBC', 'Code_Service_RES',
                    'Code_Service_SEG', 'Code_Service_ULI'
                ]
                
                df_input = pd.DataFrame(feature_data)
                # Reorder columns to match expected order exactly
                df_input = df_input[expected_columns]
                
                # Scale numerical features if scaler is available
                # Note: XGBoost can work without scaling, but scaling may improve accuracy
                numerical_features_to_scale = ['IPS', 'Taux_Mentions', 'Nb_eleves', 'Code_Postal', 'day_of_year']
                if scaler is not None:
                    try:
                        # Ensure all columns exist before scaling
                        for col in numerical_features_to_scale:
                            if col not in df_input.columns:
                                df_input[col] = 0
                        df_input[numerical_features_to_scale] = scaler.transform(df_input[numerical_features_to_scale])
                        print("[INFO] Features scaled using provided scaler")
                    except Exception as e:
                        print(f"[WARNING] Error during scaling: {e}. Proceeding without scaling.")
                else:
                    # XGBoost can work without scaling, so this is just informational
                    print("[INFO] No scaler found - using unscaled features (XGBoost handles this)")
                
                print(f"[DEBUG] Input DataFrame shape: {df_input.shape}")
                print(f"[DEBUG] Input DataFrame columns: {df_input.columns.tolist()}")
                
                # Make prediction
                prediction = xgb_model.predict(df_input)
                
                # Extract prediction value
                if isinstance(prediction, np.ndarray):
                    pred_value = float(prediction[0]) if prediction.size > 0 else float(prediction)
                else:
                    pred_value = float(prediction[0]) if len(prediction) > 0 else float(prediction)
                
                # Format as percentage with 2 decimal places
                result = f"{pred_value:.2f}%"
                
                print(f"[DEBUG] Raw prediction: {prediction}")
                print(f"[DEBUG] Formatted prediction: {result}")
                
                return result
            
            # Special handling for Taux de Mentions XGBoost Regression model
            if model_key == 'taux_mentions_xgb':
                # Load model artifacts (contains model pipeline, mlb_sec, mlb_srv, mlb_voie)
                artifacts = model
                
                # Handle different model formats
                if isinstance(artifacts, dict):
                    # Dictionary format with separate components
                    pipeline = artifacts.get('model') or artifacts.get('pipeline') or artifacts.get('best_estimator')
                    mlb_sec = artifacts.get('mlb_sec')
                    mlb_srv = artifacts.get('mlb_srv')
                    mlb_voie = artifacts.get('mlb_voie')
                    
                    if pipeline is None:
                        return f"Model pipeline not found in artifacts. Available keys: {list(artifacts.keys())}"
                else:
                    # Assume it's just the pipeline (model might be saved directly)
                    pipeline = artifacts
                    mlb_sec = None
                    mlb_srv = None
                    mlb_voie = None
                    print("[WARNING] Model loaded as pipeline only. MultiLabelBinarizers not found.")
                    print("[INFO] Attempting to use pipeline without separate MultiLabelBinarizers.")
                    print(f"[DEBUG] Model type: {type(pipeline)}")
                
                if pipeline is None:
                    return f"Model pipeline is None. Model type: {type(model)}"
                
                # Patch the pipeline
                _patch_simple_imputer(pipeline)
                
                # Process multi-label fields (Code_Section, Code_Service, Code_Voie)
                # Convert comma-separated strings to lists
                section_str = input_data.get('code_section', '').strip()
                service_str = input_data.get('code_service', '').strip()
                voie_str = input_data.get('code_voie', '').strip()
                
                section_list = [s.strip() for s in section_str.split(',') if s.strip()] if section_str else []
                service_list = [s.strip() for s in service_str.split(',') if s.strip()] if service_str else []
                voie_list = [s.strip() for s in voie_str.split(',') if s.strip()] if voie_str else []
                
                # Transform using MultiLabelBinarizers
                # Based on the error, the model expects these specific columns:
                expected_sec_cols = ['sec_THE', 'sec_ART', 'sec_SPO', 'sec_CIN', 'sec_EUR', 'sec_INT']
                expected_srv_cols = ['srv_RES', 'srv_SEG', 'srv_ULI', 'srv_GRE', 'srv_APP', 'srv_HEB', 'srv_PBC']
                expected_voie_cols = ['voie_G', 'voie_T', 'voie_P']
                
                if mlb_sec:
                    sec_dummies = pd.DataFrame(mlb_sec.transform([section_list]), 
                                             columns=[f"sec_{c}" for c in mlb_sec.classes_])
                    # Ensure all expected columns exist
                    for col in expected_sec_cols:
                        if col not in sec_dummies.columns:
                            sec_dummies[col] = 0
                    sec_dummies = sec_dummies[expected_sec_cols]  # Reorder to match expected
                else:
                    # Create all expected columns with zeros
                    sec_dummies = pd.DataFrame(0, index=[0], columns=expected_sec_cols)
                    # Set the ones that match user input
                    for sec in section_list:
                        col_name = f"sec_{sec}"
                        if col_name in expected_sec_cols:
                            sec_dummies[col_name] = 1
                
                if mlb_srv:
                    srv_dummies = pd.DataFrame(mlb_srv.transform([service_list]), 
                                             columns=[f"srv_{c}" for c in mlb_srv.classes_])
                    # Ensure all expected columns exist
                    for col in expected_srv_cols:
                        if col not in srv_dummies.columns:
                            srv_dummies[col] = 0
                    srv_dummies = srv_dummies[expected_srv_cols]  # Reorder to match expected
                else:
                    # Create all expected columns with zeros
                    srv_dummies = pd.DataFrame(0, index=[0], columns=expected_srv_cols)
                    # Set the ones that match user input
                    for srv in service_list:
                        col_name = f"srv_{srv}"
                        if col_name in expected_srv_cols:
                            srv_dummies[col_name] = 1
                
                if mlb_voie:
                    voie_dummies = pd.DataFrame(mlb_voie.transform([voie_list]), 
                                              columns=[f"voie_{c}" for c in mlb_voie.classes_])
                    # Ensure all expected columns exist
                    for col in expected_voie_cols:
                        if col not in voie_dummies.columns:
                            voie_dummies[col] = 0
                    voie_dummies = voie_dummies[expected_voie_cols]  # Reorder to match expected
                else:
                    # Create all expected columns with zeros
                    voie_dummies = pd.DataFrame(0, index=[0], columns=expected_voie_cols)
                    # Set the ones that match user input
                    for voie in voie_list:
                        col_name = f"voie_{voie}"
                        if col_name in expected_voie_cols:
                            voie_dummies[col_name] = 1
                
                # Create base DataFrame with numeric and categorical features
                base_data = {
                    'IPS': [input_data['ips']],
                    'Taux_Reussite': [input_data['taux_reussite']],
                    'Nb_eleves': [input_data['nb_eleves']],
                    'Region': [input_data['region']],
                    'Code_TypeEtab': [input_data['code_type_etab']],
                    'Statut_Etablissement': [input_data['statut_etablissement']],
                }
                
                df_input = pd.DataFrame(base_data)
                
                # Combine with multi-hot encoded columns (always add them, even if empty)
                df_input = pd.concat([df_input, sec_dummies, srv_dummies, voie_dummies], axis=1)
                
                print(f"[DEBUG] Input DataFrame shape: {df_input.shape}")
                print(f"[DEBUG] Input DataFrame columns: {df_input.columns.tolist()}")
                
                # Make prediction using the pipeline
                prediction = pipeline.predict(df_input)
                
                # Extract prediction value
                if isinstance(prediction, np.ndarray):
                    pred_value = float(prediction[0]) if prediction.size > 0 else float(prediction)
                else:
                    pred_value = float(prediction[0]) if len(prediction) > 0 else float(prediction)
                
                # Format as percentage with 2 decimal places
                result = f"{pred_value:.2f}%"
                
                print(f"[DEBUG] Raw prediction: {prediction}")
                print(f"[DEBUG] Formatted prediction: {result}")
                
                return result
            
            # Special handling for Student Number Classifier (Logistic Regression)
            if model_key == 'student_number_classifier':
                # Load model artifacts (contains model pipeline, mlb_sec, mlb_srv, mlb_voie)
                artifacts = model
                
                # Handle different model formats
                if isinstance(artifacts, dict):
                    # Dictionary format with separate components
                    pipeline = artifacts.get('model') or artifacts.get('pipeline')
                    mlb_sec = artifacts.get('mlb_sec')
                    mlb_srv = artifacts.get('mlb_srv')
                    mlb_voie = artifacts.get('mlb_voie')
                    
                    if pipeline is None:
                        return f"Model pipeline not found in artifacts. Available keys: {list(artifacts.keys())}"
                else:
                    # Assume it's just the pipeline (model might be saved directly)
                    pipeline = artifacts
                    mlb_sec = None
                    mlb_srv = None
                    mlb_voie = None
                    print("[WARNING] Model loaded as pipeline only. MultiLabelBinarizers not found.")
                    print("[INFO] Attempting to use pipeline without separate MultiLabelBinarizers.")
                    print(f"[DEBUG] Model type: {type(pipeline)}")
                
                if pipeline is None:
                    return f"Model pipeline is None. Model type: {type(model)}"
                
                # Patch the pipeline
                _patch_simple_imputer(pipeline)
                
                # Process multi-label fields (Code_Section, Code_Service, Code_Voie)
                # Convert comma-separated strings to lists
                section_str = input_data.get('code_section', '').strip()
                service_str = input_data.get('code_service', '').strip()
                voie_str = input_data.get('code_voie', '').strip()
                
                section_list = [s.strip() for s in section_str.split(',') if s.strip()] if section_str else []
                service_list = [s.strip() for s in service_str.split(',') if s.strip()] if service_str else []
                voie_list = [s.strip() for s in voie_str.split(',') if s.strip()] if voie_str else []
                
                # Transform using MultiLabelBinarizers
                if mlb_sec:
                    sec_dummies = pd.DataFrame(mlb_sec.transform([section_list]), 
                                             columns=[f"sec_{c}" for c in mlb_sec.classes_])
                else:
                    # Create empty DataFrame - will be populated based on model requirements
                    sec_dummies = pd.DataFrame()
                
                if mlb_srv:
                    srv_dummies = pd.DataFrame(mlb_srv.transform([service_list]), 
                                             columns=[f"srv_{c}" for c in mlb_srv.classes_])
                else:
                    srv_dummies = pd.DataFrame()
                
                if mlb_voie:
                    voie_dummies = pd.DataFrame(mlb_voie.transform([voie_list]), 
                                              columns=[f"voie_{c}" for c in mlb_voie.classes_])
                else:
                    voie_dummies = pd.DataFrame()
                
                # Create base DataFrame with numeric and categorical features
                base_data = {
                    'IPS': [input_data['ips']],
                    'Taux_Reussite': [input_data['taux_reussite']],
                    'Taux_Mentions': [input_data['taux_mentions']],
                    'Annee': [input_data['annee']],
                    'Region': [input_data['region']],
                    'Code_TypeEtab': [input_data['code_type_etab']],
                    'Statut_Etablissement': [input_data['statut_etablissement']],
                }
                
                df_input = pd.DataFrame(base_data)
                
                # Combine with multi-hot encoded columns
                if not sec_dummies.empty:
                    df_input = pd.concat([df_input, sec_dummies], axis=1)
                if not srv_dummies.empty:
                    df_input = pd.concat([df_input, srv_dummies], axis=1)
                if not voie_dummies.empty:
                    df_input = pd.concat([df_input, voie_dummies], axis=1)
                
                print(f"[DEBUG] Input DataFrame shape: {df_input.shape}")
                print(f"[DEBUG] Input DataFrame columns: {df_input.columns.tolist()}")
                
                # Make prediction using the pipeline
                prediction = pipeline.predict(df_input)
                prediction_proba = pipeline.predict_proba(df_input)
                
                # Extract prediction value
                if isinstance(prediction, np.ndarray):
                    pred_class = int(prediction[0]) if prediction.size > 0 else int(prediction)
                else:
                    pred_class = int(prediction[0]) if len(prediction) > 0 else int(prediction)
                
                # Get probability for class 1 (above median)
                if isinstance(prediction_proba, np.ndarray):
                    proba_class_1 = float(prediction_proba[0][1]) if prediction_proba.size > 0 else float(prediction_proba[0][1])
                else:
                    proba_class_1 = float(prediction_proba[0][1]) if len(prediction_proba) > 0 else float(prediction_proba[0][1])
                
                # Format result
                class_label = "Above Median" if pred_class == 1 else "Below Median"
                result = f"Prediction: {class_label} (Probability: {proba_class_1:.2%})"
                
                print(f"[DEBUG] Raw prediction: {prediction}")
                print(f"[DEBUG] Prediction probabilities: {prediction_proba}")
                print(f"[DEBUG] Formatted prediction: {result}")
                
                return result
            
            # Special handling for Clustering model (KMeans with PCA)
            if model_key == 'clustering':
                # Load model artifacts (contains scaler, pca_model, kmeans_model, mlb_sec, mlb_voie, mlb_srv)
                artifacts = model
                
                # Handle different model formats
                if isinstance(artifacts, dict):
                    # Dictionary format with separate components
                    scaler = artifacts.get('scaler')
                    pca = artifacts.get('pca') or artifacts.get('pca_model')
                    kmeans = artifacts.get('kmeans') or artifacts.get('kmeans_model') or artifacts.get('model')
                    mlb_sec = artifacts.get('mlb_sec')
                    mlb_voie = artifacts.get('mlb_voie')
                    mlb_srv = artifacts.get('mlb_srv')  # May or may not be used
                    
                    # Debug: Check which components are missing
                    missing = []
                    if scaler is None:
                        missing.append('scaler')
                    if pca is None:
                        missing.append('pca/pca_model')
                    if kmeans is None:
                        missing.append('kmeans/kmeans_model')
                    
                    if missing:
                        return f"Model components not found: {missing}. Available keys: {list(artifacts.keys())}"
                else:
                    return "Clustering model must be saved as a dictionary with 'scaler', 'pca_model', 'kmeans_model', 'mlb_sec', 'mlb_voie'"
                
                # Process multi-label fields (Code_Section, Code_Voie)
                # Convert comma-separated strings to lists
                section_str = input_data.get('code_section', '').strip()
                voie_str = input_data.get('code_voie', '').strip()
                
                section_list = [s.strip() for s in section_str.split(',') if s.strip()] if section_str else []
                voie_list = [s.strip() for s in voie_str.split(',') if s.strip()] if voie_str else []
                
                # Transform using MultiLabelBinarizers
                if mlb_sec:
                    sec_dummies = pd.DataFrame(mlb_sec.transform([section_list]), 
                                             columns=[f"sec_{c}" for c in mlb_sec.classes_])
                else:
                    # Create empty DataFrame - will be populated based on model requirements
                    sec_dummies = pd.DataFrame()
                
                if mlb_voie:
                    voie_dummies = pd.DataFrame(mlb_voie.transform([voie_list]), 
                                              columns=[f"voie_{c}" for c in mlb_voie.classes_])
                else:
                    voie_dummies = pd.DataFrame()
                
                # Create base DataFrame with numeric and categorical features
                base_data = {
                    'IPS': [input_data['ips']],
                    'Taux_Reussite': [input_data['taux_reussite']],
                    'Taux_Mentions': [input_data['taux_mentions']],
                    'Nb_eleves': [input_data['nb_eleves']],
                    'Code_TypeEtab': [input_data['code_type_etab']],
                }
                
                df_input = pd.DataFrame(base_data)
                
                # One-hot encode Code_TypeEtab (drop_first=True as in notebook)
                # This creates Code_TypeEtab_E and Code_TypeEtab_L (dropping Code_TypeEtab_C)
                df_input = pd.get_dummies(df_input, columns=['Code_TypeEtab'], drop_first=True)
                
                # Ensure all expected Code_TypeEtab columns exist
                # The model expects Code_TypeEtab_E and Code_TypeEtab_L (C is dropped)
                expected_type_cols = ['Code_TypeEtab_E', 'Code_TypeEtab_L']
                for col in expected_type_cols:
                    if col not in df_input.columns:
                        df_input[col] = 0
                
                # Combine with multi-hot encoded columns FIRST (before reordering)
                if not sec_dummies.empty:
                    df_input = pd.concat([df_input, sec_dummies], axis=1)
                if not voie_dummies.empty:
                    df_input = pd.concat([df_input, voie_dummies], axis=1)
                
                # Get the expected feature order from model artifacts if available
                # The artifacts contain 'numeric_features', 'categorical_features', and 'multi_hot_features'
                if 'numeric_features' in artifacts and 'multi_hot_features' in artifacts:
                    # Use the stored feature order from training
                    numeric_features = artifacts.get('numeric_features', [])
                    multi_hot_features = artifacts.get('multi_hot_features', [])
                    
                    # Build expected order: numeric + categorical (one-hot) + multi_hot
                    # Categorical features (Code_TypeEtab) are one-hot encoded as Code_TypeEtab_E, Code_TypeEtab_L
                    expected_order = []
                    
                    # Add numeric features in the stored order
                    expected_order.extend(numeric_features)
                    
                    # Add categorical one-hot features (Code_TypeEtab_E, Code_TypeEtab_L)
                    # These come after numeric but before multi_hot
                    expected_order.extend(expected_type_cols)
                    
                    # Add multi-hot features in the stored order
                    expected_order.extend(multi_hot_features)
                    
                    print(f"[DEBUG] Using stored feature order from artifacts")
                    print(f"[DEBUG] Numeric features: {numeric_features}")
                    print(f"[DEBUG] Multi-hot features: {multi_hot_features}")
                else:
                    # Fallback: construct order based on notebook structure
                    numeric_cols = ['IPS', 'Taux_Reussite', 'Taux_Mentions', 'Nb_eleves']
                    
                    # Get all sec_ columns (sorted for consistency)
                    sec_cols = sorted([c for c in df_input.columns if c.startswith('sec_')])
                    # Get all voie_ columns (sorted for consistency)
                    voie_cols = sorted([c for c in df_input.columns if c.startswith('voie_')])
                    
                    # Build the expected column order
                    expected_order = numeric_cols + expected_type_cols + sec_cols + voie_cols
                    print(f"[DEBUG] Using fallback feature order")
                
                # Ensure all columns in expected_order exist, add missing ones with zeros
                for col in expected_order:
                    if col not in df_input.columns:
                        df_input[col] = 0
                        print(f"[DEBUG] Added missing column: {col}")
                
                # Reorder to match expected order exactly
                df_input = df_input[expected_order]
                
                print(f"[DEBUG] Final DataFrame columns (in order): {df_input.columns.tolist()}")
                print(f"[DEBUG] Expected order length: {len(expected_order)}, DataFrame columns: {len(df_input.columns)}")
                
                # Check if scaler has feature_names_in_ attribute (scikit-learn 1.0+)
                # This tells us the exact order expected by the scaler
                if hasattr(scaler, 'feature_names_in_'):
                    scaler_feature_names = scaler.feature_names_in_
                    print(f"[DEBUG] Scaler expects features in this order: {scaler_feature_names.tolist()}")
                    
                    # Reorder DataFrame to match scaler's expected order
                    # Ensure all expected features exist
                    for col in scaler_feature_names:
                        if col not in df_input.columns:
                            df_input[col] = 0
                            print(f"[DEBUG] Added missing column for scaler: {col}")
                    
                    # Reorder to match scaler's expected order exactly
                    df_input = df_input[scaler_feature_names]
                    print(f"[DEBUG] Reordered DataFrame to match scaler's feature_names_in_")
                
                print(f"[DEBUG] Input DataFrame shape before scaling: {df_input.shape}")
                print(f"[DEBUG] Input DataFrame columns (final): {df_input.columns.tolist()}")
                
                # Scale the features
                X_scaled = scaler.transform(df_input)
                
                # Apply PCA
                X_pca = pca.transform(X_scaled)
                
                # Predict cluster
                cluster = kmeans.predict(X_pca)
                
                # Extract cluster number
                if isinstance(cluster, np.ndarray):
                    cluster_num = int(cluster[0]) if cluster.size > 0 else int(cluster)
                else:
                    cluster_num = int(cluster[0]) if len(cluster) > 0 else int(cluster)
                
                # Calculate distance to cluster center (optional, for additional info)
                cluster_center = kmeans.cluster_centers_[cluster_num]
                distance_to_center = np.linalg.norm(X_pca[0] - cluster_center)
                
                # Format result
                result = f"Cluster: {cluster_num} (Distance to center: {distance_to_center:.2f})"
                
                print(f"[DEBUG] Raw cluster prediction: {cluster}")
                print(f"[DEBUG] PCA components: {X_pca[0]}")
                print(f"[DEBUG] Formatted prediction: {result}")
                
                return result
            
            # Special handling for Establishment Type SVM model
            if model_key == 'establishment_type_svm':
                # Load model artifacts (contains model, mlb_sec, mlb_srv, mlb_voie)
                artifacts = model
                if artifacts is None or not isinstance(artifacts, dict):
                    return "Model artifacts not loaded correctly."
                
                pipeline = artifacts.get('model')
                mlb_sec = artifacts.get('mlb_sec')
                mlb_srv = artifacts.get('mlb_srv')
                mlb_voie = artifacts.get('mlb_voie')
                
                if pipeline is None:
                    return "Model pipeline not found in artifacts."
                
                # Patch the pipeline
                _patch_simple_imputer(pipeline)
                
                # Process multi-label fields (Code_Section, Code_Service, Code_Voie)
                # Convert comma-separated strings to lists
                section_list = [s.strip() for s in input_data['code_section'].split(',') if s.strip()] if input_data.get('code_section') else []
                service_list = [s.strip() for s in input_data['code_service'].split(',') if s.strip()] if input_data.get('code_service') else []
                voie_list = [s.strip() for s in input_data['code_voie'].split(',') if s.strip()] if input_data.get('code_voie') else []
                
                # Transform using MultiLabelBinarizers
                if mlb_sec:
                    sec_dummies = pd.DataFrame(mlb_sec.transform([section_list]), 
                                             columns=[f"sec_{c}" for c in mlb_sec.classes_])
                else:
                    sec_dummies = pd.DataFrame()
                
                if mlb_srv:
                    srv_dummies = pd.DataFrame(mlb_srv.transform([service_list]),
                                             columns=[f"srv_{c}" for c in mlb_srv.classes_])
                else:
                    srv_dummies = pd.DataFrame()
                
                if mlb_voie:
                    voie_dummies = pd.DataFrame(mlb_voie.transform([voie_list]),
                                              columns=[f"voie_{c}" for c in mlb_voie.classes_])
                else:
                    voie_dummies = pd.DataFrame()
                
                # Create base DataFrame with numeric and categorical features
                base_df = pd.DataFrame([{
                    'IPS': input_data['ips'],
                    'Taux_Reussite': input_data['taux_reussite'],
                    'Taux_Mentions': input_data['taux_mentions'],
                    'Nb_eleves': input_data['nb_eleves'],
                    'Annee': input_data['annee'],
                    'Region': input_data['region'],
                    'Statut_Etablissement': input_data['statut_etablissement']
                }])
                
                # Combine base features with multi-label binary features
                df_input = pd.concat([base_df, sec_dummies, srv_dummies, voie_dummies], axis=1)
                
                print(f"[DEBUG] Input DataFrame shape: {df_input.shape}")
                print(f"[DEBUG] Input DataFrame columns: {df_input.columns.tolist()}")
                
                # Make prediction
                prediction = pipeline.predict(df_input)
                
                # Extract prediction value (handle numpy array)
                if isinstance(prediction, np.ndarray):
                    pred_value = int(prediction[0]) if prediction.size > 0 else prediction[0]
                else:
                    pred_value = int(prediction[0]) if len(prediction) > 0 else prediction
                
                # Map numeric predictions to French names
                # 0 = Collège, 1 = École, 2 = Lycée
                class_map = {
                    0: 'Collège',
                    1: 'École',
                    2: 'Lycée'
                }
                
                result = class_map.get(pred_value, f"Unknown class ({pred_value})")
                
                print(f"[DEBUG] Raw prediction: {prediction}")
                print(f"[DEBUG] Prediction value: {pred_value}")
                print(f"[DEBUG] Decoded prediction: {result}")
                
                return result
            
            # Ensure patch is applied before prediction (safety measure) for other models
            if isinstance(model, dict) and 'model' in model:
                _patch_simple_imputer(model['model'])
            else:
                _patch_simple_imputer(model)
            
            # Special handling for Public/Private KNN model
            if model_key == 'public_private_knn':
                # Transform form data into the DataFrame structure expected by the pipeline
                date_obj = input_data['date']
                
                df_input = pd.DataFrame([{
                    'Nb_eleves': input_data['nb_eleves'],
                    'Code_Postal': input_data['code_postal'], # Pipeline expects this
                    'Region': input_data['region'],
                    'Annee': date_obj.year,
                    'Mois': date_obj.month,
                    'Jour': date_obj.day,
                    'Code_Section': input_data['code_section'],
                    'Code_Voie': input_data['code_voie'],
                    'Code_TypeEtab': input_data['code_type_etab'],
                    'Code_Service': input_data['code_service']
                }])
                
                # Ensure types match what might be expected (Code_Postal was numeric in numeric_features list of notebook)
                # But typically valid Postal Codes are strings. The user's notebook had it in numeric_features.
                # Let's try to compel it to numeric if possible, or leave as is if pipeline handles it.
                # Notebook: numeric_features = [..., 'Code_Postal', ...]
                # So we must cast it.
                try:
                    df_input['Code_Postal'] = pd.to_numeric(df_input['Code_Postal'])
                except:
                    pass # Keep as string if it fails, maybe pipeline handles it? 

                # Debug: Print input data to verify it's being processed
                print(f"[DEBUG] Input DataFrame:\n{df_input}")
                print(f"[DEBUG] Input data types:\n{df_input.dtypes}")
                
                # Verify model is actually a model object
                if not hasattr(model, 'predict'):
                    return f"Error: Model object does not have predict method. Type: {type(model)}"
                
                # Check if model has classes_ attribute (for classification models)
                if hasattr(model, 'classes_'):
                    print(f"[DEBUG] Model classes: {model.classes_}")
                
                # Make prediction
                print(f"[DEBUG] Calling model.predict() with DataFrame shape: {df_input.shape}")
                prediction = model.predict(df_input)
                print(f"[DEBUG] Raw prediction result: {prediction}")
                print(f"[DEBUG] Prediction type: {type(prediction)}")
                print(f"[DEBUG] Prediction shape: {prediction.shape if hasattr(prediction, 'shape') else 'N/A'}")
                
                # Extract the actual prediction value
                if isinstance(prediction, np.ndarray):
                    pred_value = prediction.item() if prediction.size == 1 else prediction[0]
                elif isinstance(prediction, (list, tuple)):
                    pred_value = prediction[0] if len(prediction) > 0 else prediction
                else:
                    pred_value = prediction
                
                # Convert to string if needed and ensure it's a proper value
                if isinstance(pred_value, (np.integer, np.floating)):
                    pred_value = pred_value.item()
                
                print(f"[DEBUG] Final prediction value: {pred_value} (type: {type(pred_value)})")
                
                # Verify the prediction is not None or empty
                if pred_value is None:
                    return "Error: Model returned None prediction"
                
                return str(pred_value)

            # Generic fallback for other models
            if isinstance(input_data, dict):
                df = pd.DataFrame([input_data])
                prediction = model.predict(df)
            else:
                prediction = model.predict([input_data])
                
            return prediction[0]
        except Exception as e:
            return f"Error during prediction: {str(e)}"

# Clear cache on module import to ensure models are reloaded with patches
# This happens automatically when Django restarts
ModelLoader.clear_cache()
