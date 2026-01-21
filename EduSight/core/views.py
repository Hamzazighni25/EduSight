from django.shortcuts import render
from django.views.decorators.clickjacking import xframe_options_sameorigin
from .ml_models.loader import ModelLoader
from .forms import PredictionForm

def home(request):
    return render(request, 'home.html')

@xframe_options_sameorigin
def dashboard(request):
    """
    Renders the Power BI dashboard.
    """
    # Power BI embed URL
    powerbi_url = "https://app.powerbi.com/reportEmbed?reportId=eb03af80-0fe2-4efb-90c4-000b1972fe44&autoAuth=true&ctid=604f1a96-cbe8-43f8-abbf-f8eaf5d85730"
    
    context = {
        'powerbi_url': powerbi_url
    }
    return render(request, 'dashboard.html', context)

def objectives(request):
    """
    Display ML objectives and their corresponding models.
    """
    objectives_data = {
        'main_title': 'Optimize the distribution and quality of schools for equitable education',
        'sections': [
            {
                'title': 'Improve the distribution and structure of schools',
                'objectives': [
                    {
                        'number': '1.1',
                        'title': 'Optimize geographical distribution of schools',
                        'description': 'Identify areas with too many or too few schools to support better territorial balance.',
                        'model': 'K-Means clustering',
                        'model_key': 'clustering',
                        'icon': 'fa-map-marked-alt'
                    },
                    {
                        'number': '1.2',
                        'title': 'Anticipate student demand',
                        'description': 'Forecast student numbers to better plan infrastructure, classrooms, and teaching staff.',
                        'model': 'Supervised learning model (Logistic Regression)',
                        'model_key': 'student_number_classifier',
                        'icon': 'fa-users'
                    },
                    {
                        'number': '1.3',
                        'title': 'Improve school type planning',
                        'description': 'Identify the most appropriate school level needed in each area (primary, middle, or high school).',
                        'model': 'Multi-class classification model (SVC)',
                        'model_key': 'establishment_type_svm',
                        'icon': 'fa-school'
                    }
                ]
            },
            {
                'title': 'Improve school quality and performance',
                'objectives': [
                    {
                        'number': '2.1',
                        'title': 'Improve academic excellence',
                        'description': 'Anticipate academic performance levels to support improvement strategies.',
                        'model': 'XGBoost Regressor',
                        'model_key': 'taux_mentions_xgb',
                        'icon': 'fa-trophy'
                    },
                    {
                        'number': '2.2',
                        'title': 'Analyze public vs private education',
                        'description': 'Better understand the distribution of public and private schools to support strategic analysis.',
                        'model': 'K-Nearest Neighbors (KNN)',
                        'model_key': 'public_private_knn',
                        'icon': 'fa-balance-scale'
                    },
                    {
                        'number': '2.3',
                        'title': 'Improve overall success rates',
                        'description': 'Identify schools that may need performance improvement actions.',
                        'model': 'XGBoost Regressor',
                        'model_key': 'taux_reussite_xgb',
                        'icon': 'fa-chart-line'
                    }
                ]
            }
        ]
    }
    
    return render(request, 'objectives.html', {'objectives': objectives_data})


def predict(request, model_name='public_private_knn'):
    """
    Generic view to handle predictions for different models.
    """
    result = None
    
    # Select the correct form based on the model
    form_class = PredictionForm # Default
    if model_name == 'public_private_knn':
        from .forms import PublicPrivateForm
        form_class = PublicPrivateForm
    elif model_name == 'establishment_type_svm':
        from .forms import EstablishmentTypeForm
        form_class = EstablishmentTypeForm
    elif model_name == 'taux_reussite_xgb':
        from .forms import TauxReussiteForm
        form_class = TauxReussiteForm
    elif model_name == 'taux_mentions_xgb':
        from .forms import TauxMentionsForm
        form_class = TauxMentionsForm
    elif model_name == 'student_number_classifier':
        from .forms import StudentNumberForm
        form_class = StudentNumberForm
    elif model_name == 'clustering':
        from .forms import ClusteringForm
        form_class = ClusteringForm
    
    form = form_class(request.POST or None)
    
    if request.method == 'POST':
        if form.is_valid():
            # Extract data from form
            input_data = form.cleaned_data
            
            # Run prediction
            result = ModelLoader.predict(model_name, input_data)

    context = {
        'form': form,
        'result': result,
        'model_name': model_name,
        'available_models': ModelLoader.MODEL_FILES.keys()
    }
    return render(request, 'predict.html', context)
