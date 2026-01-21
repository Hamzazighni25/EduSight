from django import forms
import datetime

class PredictionForm(forms.Form):
    """
    Generic fallback form.
    """
    feature1 = forms.FloatField(label='Feature 1', required=True)
    feature2 = forms.FloatField(label='Feature 2', required=True)

class PublicPrivateForm(forms.Form):
    """
    Form specifically for the KNN Public/Private Classifier.
    Features based on user's notebook:
    Nb_eleves, Code_Postal, Region, Code_Section, Code_Voie, Code_TypeEtab, Code_Service, Date(Annee, Mois, Jour)
    """
    nb_eleves = forms.IntegerField(
        label='Number of Students (Nb_eleves)', 
        min_value=0, 
        initial=100,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter number of students'})
    )
    code_postal = forms.CharField(
        label='Postal Code (Code_Postal)', 
        max_length=10, 
        initial='75001',
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 75001'})
    )
    region = forms.CharField(
        label='Region', 
        max_length=100, 
        initial='Île-de-France',
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g., Île-de-France'})
    )
    
    # Using a DateField to easily extract Annee, Mois, Jour
    date = forms.DateField(
        label='Date (Extracts Year, Month, Day)', 
        widget=forms.DateInput(attrs={'type': 'date', 'class': 'form-control'}),
        initial=datetime.date.today
    )
    
    code_section = forms.CharField(
        label='Section Code (Code_Section)', 
        max_length=50, 
        initial='Unknown', 
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Unknown (Optional)'})
    )
    code_voie = forms.CharField(
        label='Track Code (Code_Voie)', 
        max_length=50, 
        initial='Unknown', 
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Unknown (Optional)'})
    )
    code_type_etab = forms.CharField(
        label='Establishment Type Code (Code_TypeEtab)', 
        max_length=50, 
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Optional'})
    )
    code_service = forms.CharField(
        label='Service Code (Code_Service)', 
        max_length=50, 
        initial='Unknown', 
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Unknown (Optional)'})
    )

    def clean(self):
        cleaned_data = super().clean()
        # Fill defaults if left empty for non-required fields exactly like the training cleaning step
        if not cleaned_data.get('code_section'): cleaned_data['code_section'] = 'Unknown'
        if not cleaned_data.get('code_voie'): cleaned_data['code_voie'] = 'Unknown'
        if not cleaned_data.get('code_service'): cleaned_data['code_service'] = 'Unknown'
        return cleaned_data

class EstablishmentTypeForm(forms.Form):
    """
    Form for SVM Establishment Type Classifier (École/Collège/Lycée).
    Features: IPS, Taux_Reussite, Taux_Mentions, Nb_eleves, Region, Statut_Etablissement,
    Code_Section, Code_Service, Code_Voie, Annee
    """
    ips = forms.FloatField(
        label='IPS (Indice de Position Sociale)',
        min_value=0,
        max_value=200,
        initial=100,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 100', 'step': '0.1'})
    )
    taux_reussite = forms.FloatField(
        label='Taux de Réussite (%)',
        min_value=0,
        max_value=100,
        initial=80.0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 80.0', 'step': '0.1'})
    )
    taux_mentions = forms.FloatField(
        label='Taux de Mentions (%)',
        min_value=0,
        max_value=100,
        initial=30.0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 30.0', 'step': '0.1'})
    )
    nb_eleves = forms.IntegerField(
        label='Number of Students (Nb_eleves)',
        min_value=0,
        initial=500,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter number of students'})
    )
    region = forms.CharField(
        label='Region',
        max_length=100,
        initial='Île-de-France',
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g., Île-de-France'})
    )
    statut_etablissement = forms.ChoiceField(
        label='Statut Établissement',
        choices=[('Public', 'Public'), ('Privé', 'Privé')],
        initial='Public',
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    code_section = forms.CharField(
        label='Section Code(s) (Code_Section) - Comma-separated if multiple',
        max_length=200,
        initial='',
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g., S, ES or leave empty'})
    )
    code_service = forms.CharField(
        label='Service Code(s) (Code_Service) - Comma-separated if multiple',
        max_length=200,
        initial='',
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 1, 2 or leave empty'})
    )
    code_voie = forms.CharField(
        label='Track Code(s) (Code_Voie) - Comma-separated if multiple',
        max_length=200,
        initial='',
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g., GT, PRO or leave empty'})
    )
    annee = forms.IntegerField(
        label='Year (Annee)',
        initial=2024,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 2024'})
    )

    def clean(self):
        cleaned_data = super().clean()
        # Ensure empty multi-label fields are set to empty string (will be handled as empty list)
        if not cleaned_data.get('code_section'):
            cleaned_data['code_section'] = ''
        if not cleaned_data.get('code_service'):
            cleaned_data['code_service'] = ''
        if not cleaned_data.get('code_voie'):
            cleaned_data['code_voie'] = ''
        return cleaned_data

class TauxReussiteForm(forms.Form):
    """
    Form for XGBoost Regression model predicting Taux de Réussite (Success Rate).
    Features: IPS, Taux_Mentions, Nb_eleves, Code_Postal, Region, Statut_Etablissement,
    Code_TypeEtab, Code_Section, Code_Voie, Code_Service, Date (for temporal features)
    """
    ips = forms.FloatField(
        label='IPS (Indice de Position Sociale)',
        min_value=0,
        max_value=200,
        initial=100,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 100', 'step': '0.1'})
    )
    taux_mentions = forms.FloatField(
        label='Taux de Mentions (%)',
        min_value=0,
        max_value=100,
        initial=30.0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 30.0', 'step': '0.1'})
    )
    nb_eleves = forms.IntegerField(
        label='Number of Students (Nb_eleves)',
        min_value=0,
        initial=500,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter number of students'})
    )
    code_postal = forms.CharField(
        label='Postal Code (Code_Postal)',
        max_length=10,
        initial='75001',
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 75001'})
    )
    region = forms.ChoiceField(
        label='Region',
        choices=[
            ('Auvergne-Rhône-Alpes', 'Auvergne-Rhône-Alpes'),
            ('Bourgogne-Franche-Comté', 'Bourgogne-Franche-Comté'),
            ('Bretagne', 'Bretagne'),
            ('Centre-Val de Loire', 'Centre-Val de Loire'),
            ('Corse', 'Corse'),
            ('Grand Est', 'Grand Est'),
            ('Guadeloupe', 'Guadeloupe'),
            ('Guyane', 'Guyane'),
            ('Hauts-de-France', 'Hauts-de-France'),
            ('Ile-de-France', 'Ile-de-France'),
            ('La Réunion', 'La Réunion'),
            ('Martinique', 'Martinique'),
            ('Mayotte', 'Mayotte'),
            ('Normandie', 'Normandie'),
            ('Nouvelle-Aquitaine', 'Nouvelle-Aquitaine'),
            ('Occitanie', 'Occitanie'),
            ('Pays de la Loire', 'Pays de la Loire'),
            ("Provence-Alpes-Côte d'Azur", "Provence-Alpes-Côte d'Azur"),
            ('TOM et Collectivités territoriales', 'TOM et Collectivités territoriales'),
        ],
        initial='Ile-de-France',
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    statut_etablissement = forms.ChoiceField(
        label='Statut Établissement',
        choices=[('Public', 'Public'), ('Privé', 'Privé')],
        initial='Public',
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    code_type_etab = forms.ChoiceField(
        label='Type d\'Établissement (Code_TypeEtab)',
        choices=[('C', 'Collège'), ('E', 'École'), ('L', 'Lycée')],
        initial='C',
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    code_section = forms.ChoiceField(
        label='Section Code (Code_Section)',
        choices=[
            ('', '-- Select or leave empty --'),
            ('SPO', 'SPO'),
            ('ART', 'ART'),
            ('CIN', 'CIN'),
            ('EUR', 'EUR'),
            ('INT', 'INT'),
            ('THE', 'THE'),
        ],
        required=False,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    code_voie = forms.ChoiceField(
        label='Track Code (Code_Voie)',
        choices=[
            ('', '-- Select or leave empty --'),
            ('G', 'G'),
            ('P', 'P'),
            ('T', 'T'),
        ],
        required=False,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    code_service = forms.ChoiceField(
        label='Service Code (Code_Service)',
        choices=[
            ('', '-- Select or leave empty --'),
            ('APP', 'APP'),
            ('GRE', 'GRE'),
            ('HEB', 'HEB'),
            ('PBC', 'PBC'),
            ('RES', 'RES'),
            ('SEG', 'SEG'),
            ('ULI', 'ULI'),
        ],
        required=False,
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    date = forms.DateField(
        label='Date (for temporal features)',
        widget=forms.DateInput(attrs={'type': 'date', 'class': 'form-control'}),
        initial=datetime.date.today
    )

    def clean(self):
        cleaned_data = super().clean()
        # Fill defaults for optional fields
        if not cleaned_data.get('code_section'):
            cleaned_data['code_section'] = 'Missing'
        if not cleaned_data.get('code_voie'):
            cleaned_data['code_voie'] = 'Missing'
        if not cleaned_data.get('code_service'):
            cleaned_data['code_service'] = 'Missing'
        return cleaned_data


class TauxMentionsForm(forms.Form):
    """
    Form for XGBoost Regression model predicting Taux de Mentions (Mention Rate).
    Features: IPS, Taux_Reussite, Nb_eleves, Region, Statut_Etablissement,
    Code_TypeEtab, Code_Section (multi-label), Code_Voie (multi-label), Code_Service (multi-label)
    """
    ips = forms.FloatField(
        label='IPS (Indice de Position Sociale)',
        min_value=0,
        max_value=200,
        initial=100,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 100', 'step': '0.1'})
    )
    taux_reussite = forms.FloatField(
        label='Taux de Réussite (%)',
        min_value=0,
        max_value=100,
        initial=85.0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 85.0', 'step': '0.1'})
    )
    nb_eleves = forms.IntegerField(
        label='Number of Students (Nb_eleves)',
        min_value=0,
        initial=500,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter number of students'})
    )
    region = forms.ChoiceField(
        label='Region',
        choices=[
            ('Auvergne-Rhône-Alpes', 'Auvergne-Rhône-Alpes'),
            ('Bourgogne-Franche-Comté', 'Bourgogne-Franche-Comté'),
            ('Bretagne', 'Bretagne'),
            ('Centre-Val de Loire', 'Centre-Val de Loire'),
            ('Corse', 'Corse'),
            ('Grand Est', 'Grand Est'),
            ('Guadeloupe', 'Guadeloupe'),
            ('Guyane', 'Guyane'),
            ('Hauts-de-France', 'Hauts-de-France'),
            ('Ile-de-France', 'Ile-de-France'),
            ('La Réunion', 'La Réunion'),
            ('Martinique', 'Martinique'),
            ('Mayotte', 'Mayotte'),
            ('Normandie', 'Normandie'),
            ('Nouvelle-Aquitaine', 'Nouvelle-Aquitaine'),
            ('Occitanie', 'Occitanie'),
            ('Pays de la Loire', 'Pays de la Loire'),
            ("Provence-Alpes-Côte d'Azur", "Provence-Alpes-Côte d'Azur"),
            ('TOM et Collectivités territoriales', 'TOM et Collectivités territoriales'),
        ],
        initial='Ile-de-France',
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    statut_etablissement = forms.ChoiceField(
        label='Statut Établissement',
        choices=[('Public', 'Public'), ('Privé', 'Privé')],
        initial='Public',
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    code_type_etab = forms.ChoiceField(
        label='Type d\'Établissement (Code_TypeEtab)',
        choices=[('C', 'Collège'), ('E', 'École'), ('L', 'Lycée')],
        initial='C',
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    code_section = forms.CharField(
        label='Section Code(s) (Code_Section) - comma-separated for multiple',
        max_length=200,
        initial='',
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g., SPO,ART or leave empty'})
    )
    code_voie = forms.CharField(
        label='Track Code(s) (Code_Voie) - comma-separated for multiple',
        max_length=200,
        initial='',
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g., G,P or leave empty'})
    )
    code_service = forms.CharField(
        label='Service Code(s) (Code_Service) - comma-separated for multiple',
        max_length=200,
        initial='',
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g., APP,GRE or leave empty'})
    )

    def clean(self):
        cleaned_data = super().clean()
        # Multi-label fields can be empty or comma-separated
        # They will be converted to lists in the prediction logic
        return cleaned_data


class ClusteringForm(forms.Form):
    """
    Form for KMeans Clustering model (with PCA) for grouping establishments.
    Features: IPS, Taux_Reussite, Taux_Mentions, Nb_eleves, Code_TypeEtab,
    Code_Section (multi-label), Code_Voie (multi-label)
    """
    ips = forms.FloatField(
        label='IPS (Indice de Position Sociale)',
        min_value=0,
        max_value=200,
        initial=100,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 100', 'step': '0.1'})
    )
    taux_reussite = forms.FloatField(
        label='Taux de Réussite (%)',
        min_value=0,
        max_value=100,
        initial=85.0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 85.0', 'step': '0.1'})
    )
    taux_mentions = forms.FloatField(
        label='Taux de Mentions (%)',
        min_value=0,
        max_value=100,
        initial=30.0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 30.0', 'step': '0.1'})
    )
    nb_eleves = forms.IntegerField(
        label='Number of Students (Nb_eleves)',
        min_value=0,
        initial=500,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'Enter number of students'})
    )
    code_type_etab = forms.ChoiceField(
        label='Type d\'Établissement (Code_TypeEtab)',
        choices=[('C', 'Collège'), ('E', 'École'), ('L', 'Lycée')],
        initial='C',
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    code_section = forms.CharField(
        label='Section Code(s) (Code_Section) - comma-separated for multiple',
        max_length=200,
        initial='',
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g., SPO,ART or leave empty'})
    )
    code_voie = forms.CharField(
        label='Track Code(s) (Code_Voie) - comma-separated for multiple',
        max_length=200,
        initial='',
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g., G,P or leave empty'})
    )

    def clean(self):
        cleaned_data = super().clean()
        # Multi-label fields can be empty or comma-separated
        # They will be converted to lists in the prediction logic
        return cleaned_data


class StudentNumberForm(forms.Form):
    """
    Form for Logistic Regression Classifier predicting if number of students is above/below median.
    Features: IPS, Taux_Reussite, Taux_Mentions, Annee, Region, Statut_Etablissement,
    Code_TypeEtab, Code_Section (multi-label), Code_Voie (multi-label), Code_Service (multi-label)
    """
    ips = forms.FloatField(
        label='IPS (Indice de Position Sociale)',
        min_value=0,
        max_value=200,
        initial=100,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 100', 'step': '0.1'})
    )
    taux_reussite = forms.FloatField(
        label='Taux de Réussite (%)',
        min_value=0,
        max_value=100,
        initial=85.0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 85.0', 'step': '0.1'})
    )
    taux_mentions = forms.FloatField(
        label='Taux de Mentions (%)',
        min_value=0,
        max_value=100,
        initial=30.0,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 30.0', 'step': '0.1'})
    )
    annee = forms.IntegerField(
        label='Année (Annee)',
        min_value=2000,
        max_value=2100,
        initial=2024,
        widget=forms.NumberInput(attrs={'class': 'form-control', 'placeholder': 'e.g., 2024'})
    )
    region = forms.ChoiceField(
        label='Region',
        choices=[
            ('Auvergne-Rhône-Alpes', 'Auvergne-Rhône-Alpes'),
            ('Bourgogne-Franche-Comté', 'Bourgogne-Franche-Comté'),
            ('Bretagne', 'Bretagne'),
            ('Centre-Val de Loire', 'Centre-Val de Loire'),
            ('Corse', 'Corse'),
            ('Grand Est', 'Grand Est'),
            ('Guadeloupe', 'Guadeloupe'),
            ('Guyane', 'Guyane'),
            ('Hauts-de-France', 'Hauts-de-France'),
            ('Ile-de-France', 'Ile-de-France'),
            ('La Réunion', 'La Réunion'),
            ('Martinique', 'Martinique'),
            ('Mayotte', 'Mayotte'),
            ('Normandie', 'Normandie'),
            ('Nouvelle-Aquitaine', 'Nouvelle-Aquitaine'),
            ('Occitanie', 'Occitanie'),
            ('Pays de la Loire', 'Pays de la Loire'),
            ("Provence-Alpes-Côte d'Azur", "Provence-Alpes-Côte d'Azur"),
            ('TOM et Collectivités territoriales', 'TOM et Collectivités territoriales'),
        ],
        initial='Ile-de-France',
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    statut_etablissement = forms.ChoiceField(
        label='Statut Établissement',
        choices=[('Public', 'Public'), ('Privé', 'Privé')],
        initial='Public',
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    code_type_etab = forms.ChoiceField(
        label='Type d\'Établissement (Code_TypeEtab)',
        choices=[('C', 'Collège'), ('E', 'École'), ('L', 'Lycée')],
        initial='C',
        widget=forms.Select(attrs={'class': 'form-control'})
    )
    code_section = forms.CharField(
        label='Section Code(s) (Code_Section) - comma-separated for multiple',
        max_length=200,
        initial='',
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g., SPO,ART or leave empty'})
    )
    code_voie = forms.CharField(
        label='Track Code(s) (Code_Voie) - comma-separated for multiple',
        max_length=200,
        initial='',
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g., G,P or leave empty'})
    )
    code_service = forms.CharField(
        label='Service Code(s) (Code_Service) - comma-separated for multiple',
        max_length=200,
        initial='',
        required=False,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'e.g., APP,GRE or leave empty'})
    )

    def clean(self):
        cleaned_data = super().clean()
        # Multi-label fields can be empty or comma-separated
        # They will be converted to lists in the prediction logic
        return cleaned_data
