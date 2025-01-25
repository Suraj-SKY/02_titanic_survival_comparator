import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

# Custom classifier for model selection
class RandomModelSelector(BaseEstimator, ClassifierMixin):
    def __init__(self, models):
        self.models = models
        self.selected_model = None
        self.classes_ = None  # Add classes_ attribute
        
    def fit(self, X, y):
        self.selected_model = np.random.choice(self.models)
        self.selected_model.fit(X, y)
        self.classes_ = self.selected_model.classes_  # Set classes from selected model
        return self
    
    def predict(self, X):
        return self.selected_model.predict(X)
    
    def predict_proba(self, X):
        return self.selected_model.predict_proba(X)
    
    def get_params(self, deep=True):
        return {'models': self.models}
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

# Data preprocessing and feature engineering
def load_data():
    df = pd.read_csv(r"data/Titanic.csv")
    
    # Feature engineering
    df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 
                                     'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace('Mlle', 'Miss')
    df['Title'] = df['Title'].replace('Ms', 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')
    
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = 1
    df.loc[df['FamilySize'] > 1, 'IsAlone'] = 0
    
    # Drop unnecessary columns
    df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
    
    # Define features and target
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    
    # Create preprocessing pipeline
    numeric_features = ['Age', 'Fare', 'FamilySize', 'SibSp', 'Parch']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    
    categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title']
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])
    
    X_processed = preprocessor.fit_transform(X)
    
    return X_processed, y, preprocessor

# Decision boundary visualization using PCA
def plot_decision_boundary(model, X, y, title, ax):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X.toarray() if hasattr(X, "toarray") else X)
    
    h = 0.5
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdYlBu)
    ax.set_title(title)
    ax.set_xlabel('PCA Component 1')
    ax.set_ylabel('PCA Component 2')

# Load data
X, y, preprocessor = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Streamlit app
st.title("⛴️ Titanic Survival Classifier Comparator")
st.markdown("Compare individual models, bagged versions, and combined ensembles")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.subheader("Individual Models")
    selected_models = st.multiselect(
        "Select base models:",
        ["Decision Tree", "Logistic Regression", "SVM", "KNN"],
        default=["Decision Tree"]
    )
    
    st.subheader("🎚️ Bagging Parameters")
    n_estimators = st.number_input("Number of estimators:", min_value=1, value=50)
    max_samples = st.slider("Max samples ratio:", 0.1, 1.0, 0.8)
    bootstrap_samples = st.checkbox("Bootstrap samples", value=True)
    max_features = st.slider("Max features:", 1, X.shape[1], X.shape[1]//2)
    bootstrap_features = st.checkbox("Bootstrap features", value=False)
    
    st.subheader("🤖 Combined Ensemble")
    ensemble_models = st.multiselect(
        "Select models for ensemble:",
        ["Decision Tree", "Logistic Regression", "SVM", "KNN"],
        default=["Decision Tree", "Logistic Regression"]
    )
    ensemble_estimators = st.number_input("Ensemble estimators:", min_value=1, value=30)

# Main content
if st.button("🚀 Run Analysis"):
    if not selected_models and not ensemble_models:
        st.error("Please select at least one model or ensemble!")
    else:
        progress_bar = st.progress(0)
        results = []
        ensemble_results = []
        total_steps = len(selected_models) + (1 if ensemble_models else 0)
        current_step = 0
        
        # Individual models analysis
        if selected_models:
            st.subheader("🔬 Individual Model Analysis")
            for model_name in selected_models:
                current_step += 1
                progress_bar.progress(current_step/total_steps)
                
                col1, col2 = st.columns(2)
                model_map = {
                    "Decision Tree": DecisionTreeClassifier(random_state=42),
                    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                    "SVM": SVC(probability=True, random_state=42),
                    "KNN": KNeighborsClassifier()
                }
                base_model = model_map[model_name]

                with col1:
                    st.subheader(f"🌳 {model_name} (Base)")
                    base_model.fit(X_train, y_train)
                    y_pred = base_model.predict(X_test)
                    
                    # Metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred)
                    recall = recall_score(y_test, y_pred)
                    f1 = f1_score(y_test, y_pred)
                    try:
                        roc_auc = roc_auc_score(y_test, base_model.predict_proba(X_test)[:,1])
                    except:
                        roc_auc = 0.5
                    
                    st.metric("Accuracy", f"{accuracy:.2%}")
                    st.metric("Precision", f"{precision:.2%}")
                    st.metric("Recall", f"{recall:.2%}")
                    st.metric("F1-Score", f"{f1:.2%}")
                    st.metric("ROC AUC", f"{roc_auc:.2%}")
                    
                    # Confusion matrix
                    st.markdown("**Confusion Matrix**")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    st.pyplot(fig)
                    
                    # Decision boundary
                    st.markdown("**Decision Boundary**")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    plot_decision_boundary(base_model, X_train, y_train, 
                                         f"Base {model_name}", ax)
                    st.pyplot(fig)

                with col2:
                    st.subheader(f"📦 {model_name} (Bagged)")
                    bagging_model = BaggingClassifier(
                        estimator=base_model,
                        n_estimators=n_estimators,
                        max_samples=max_samples,
                        bootstrap=bootstrap_samples,
                        max_features=max_features,
                        bootstrap_features=bootstrap_features,
                        random_state=42
                    )
                    
                    bagging_model.fit(X_train, y_train)
                    y_pred_bag = bagging_model.predict(X_test)
                    
                    # Metrics
                    accuracy_bag = accuracy_score(y_test, y_pred_bag)
                    precision_bag = precision_score(y_test, y_pred_bag)
                    recall_bag = recall_score(y_test, y_pred_bag)
                    f1_bag = f1_score(y_test, y_pred_bag)
                    try:
                        roc_auc_bag = roc_auc_score(y_test, bagging_model.predict_proba(X_test)[:,1])
                    except:
                        roc_auc_bag = 0.5
                    
                    st.metric("Accuracy", f"{accuracy_bag:.2%}")
                    st.metric("Precision", f"{precision_bag:.2%}")
                    st.metric("Recall", f"{recall_bag:.2%}")
                    st.metric("F1-Score", f"{f1_bag:.2%}")
                    st.metric("ROC AUC", f"{roc_auc_bag:.2%}")
                    
                    # Confusion matrix
                    st.markdown("**Confusion Matrix**")
                    cm_bag = confusion_matrix(y_test, y_pred_bag)
                    fig, ax = plt.subplots()
                    sns.heatmap(cm_bag, annot=True, fmt='d', cmap='Oranges', ax=ax)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    st.pyplot(fig)
                    
                    # Decision boundary
                    st.markdown("**Decision Boundary**")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    plot_decision_boundary(bagging_model, X_train, y_train,
                                         f"Bagged {model_name}", ax)
                    st.pyplot(fig)

                results.append({
                    'Model': model_name,
                    'Base Accuracy': accuracy,
                    'Bagged Accuracy': accuracy_bag,
                    'Base F1': f1,
                    'Bagged F1': f1_bag,
                    'Base ROC AUC': roc_auc,
                    'Bagged ROC AUC': roc_auc_bag
                })

        # Combined ensemble analysis
        if ensemble_models:
            current_step += 1
            progress_bar.progress(current_step/total_steps)
            
            st.subheader("🤖 Combined Bagging Ensemble")
            col1, col2 = st.columns(2)
            
            # Create base models
            model_map = {
                "Decision Tree": DecisionTreeClassifier(random_state=42),
                "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
                "SVM": SVC(probability=True, random_state=42),
                "KNN": KNeighborsClassifier()
            }
            selected_base_models = [model_map[name] for name in ensemble_models]
            
            # Create custom ensemble
            ensemble = BaggingClassifier(
                estimator=RandomModelSelector(selected_base_models),
                n_estimators=ensemble_estimators,
                max_samples=max_samples,
                bootstrap=bootstrap_samples,
                max_features=max_features,
                bootstrap_features=bootstrap_features,
                random_state=42
            )
            
            with col1:
                st.markdown("### 🧩 Ensemble Configuration")
                st.write(f"**Selected models:** {', '.join(ensemble_models)}")
                st.write(f"**Total estimators:** {ensemble_estimators}")
                st.write(f"**Max samples:** {max_samples}")
                st.write(f"**Bootstrap samples:** {bootstrap_samples}")
                st.write(f"**Max features:** {max_features}")
                st.write(f"**Bootstrap features:** {bootstrap_features}")
            
            with col2:
                st.markdown("### 📊 Ensemble Performance")
                ensemble.fit(X_train, y_train)
                y_pred_ens = ensemble.predict(X_test)
                
                # Metrics
                accuracy_ens = accuracy_score(y_test, y_pred_ens)
                precision_ens = precision_score(y_test, y_pred_ens)
                recall_ens = recall_score(y_test, y_pred_ens)
                f1_ens = f1_score(y_test, y_pred_ens)
                try:
                    roc_auc_ens = roc_auc_score(y_test, ensemble.predict_proba(X_test)[:,1])
                except:
                    roc_auc_ens = 0.5
                
                st.metric("Accuracy", f"{accuracy_ens:.2%}")
                st.metric("Precision", f"{precision_ens:.2%}")
                st.metric("Recall", f"{recall_ens:.2%}")
                st.metric("F1-Score", f"{f1_ens:.2%}")
                st.metric("ROC AUC", f"{roc_auc_ens:.2%}")
                
                # Confusion matrix
                st.markdown("**Confusion Matrix**")
                cm_ens = confusion_matrix(y_test, y_pred_ens)
                fig, ax = plt.subplots()
                sns.heatmap(cm_ens, annot=True, fmt='d', cmap='Purples', ax=ax)
                ax.set_xlabel('Predicted')
                ax.set_ylabel('Actual')
                st.pyplot(fig)
                
                # Decision boundary
                st.markdown("**Decision Boundary**")
                fig, ax = plt.subplots(figsize=(8, 6))
                plot_decision_boundary(ensemble, X_train, y_train,
                                     "Combined Ensemble", ax)
                st.pyplot(fig)

            ensemble_results.append({
                'Model': 'Combined Ensemble',
                'Accuracy': accuracy_ens,
                'Precision': precision_ens,
                'Recall': recall_ens,
                'F1-Score': f1_ens,
                'ROC AUC': roc_auc_ens
            })

        # Display results
        st.success("✅ Analysis Complete!")
        if results:
            st.subheader("📈 Individual Model Results")
            summary_df = pd.DataFrame(results)
            st.dataframe(summary_df.style.format({
                'Base Accuracy': '{:.2%}',
                'Bagged Accuracy': '{:.2%}',
                'Base F1': '{:.2%}',
                'Bagged F1': '{:.2%}',
                'Base ROC AUC': '{:.2%}',
                'Bagged ROC AUC': '{:.2%}'
            }))
        
        if ensemble_results:
            st.subheader("🚀 Ensemble Results")
            ensemble_df = pd.DataFrame(ensemble_results)
            st.dataframe(ensemble_df.style.format({
                'Accuracy': '{:.2%}',
                'Precision': '{:.2%}',
                'Recall': '{:.2%}',
                'F1-Score': '{:.2%}',
                'ROC AUC': '{:.2%}'
            }))

# Data exploration section
st.markdown("---")
st.subheader("🔍 Data Exploration")
if st.checkbox("Show survival distribution analysis"):
    df = pd.read_csv(r"data/Titanic.csv")
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    
    # Survival by Class
    sns.countplot(x='Pclass', hue='Survived', data=df, ax=ax[0], palette='viridis')
    ax[0].set_title('Survival by Passenger Class')
    ax[0].set_xlabel('Passenger Class')
    ax[0].set_ylabel('Count')
    
    # Survival by Gender
    sns.countplot(x='Sex', hue='Survived', data=df, ax=ax[1], palette='magma')
    ax[1].set_title('Survival by Gender')
    ax[1].set_xlabel('Gender')
    ax[1].set_ylabel('Count')
    
    st.pyplot(fig)