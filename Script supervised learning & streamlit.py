import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import streamlit as st
import numpy as np
import time

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

# Title of the Streamlit app
st.title("Klasifikasi Supervised Learning")

# Sidebar dengan tiga opsi
option = st.sidebar.selectbox(
    'Pilih opsi di sidebar',
    ('Upload CSV', 'Penjelasan', 'Report', 'Pembandingan', 'Hasil')
)

# Sidebar pertama: Upload CSV
if option == 'Upload CSV':
    st.sidebar.subheader('Upload CSV File')
    data_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

    if data_file is not None:
        data = pd.read_csv(data_file)
        st.write("Loaded Data:")
        st.write(data)

        # Encode categorical features
        data = data.apply(LabelEncoder().fit_transform)

        # Identify the target variable
        target_variable = 'Category'

        # Identify the features (columns) to be used for prediction
        features = ['Event Name','Low Level Category', 'Log Source', 'Time', 'Source IP', 'Event Count', 'Source Port', 'Destination']

        
        # Split the data into features (X) and target (y)
        X = data[features]
        y = data[target_variable]

        # Check the mapping of the LabelEncoder
        label_encoder = LabelEncoder()
        label_encoder.fit(data['Category'])
        label_mapping = {int(k): int(v) for k, v in zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))}
        st.write("Label Mapping:")
        st.write(label_mapping)

        # Split the data into training and testing sets with all features
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Dropdown menu for classifier selection
        classifier_name = st.selectbox(
            'Select Classifier',
            ('Decision Tree', 'SVM', 'KNN', 'Logistic Regression', 'MLP')
        )

        # Slider for hyperparameter tuning
        param = st.slider("Select parameter for classifier", 1, 1000, 535)
        
        # Initialize the classifier based on user selection
        if classifier_name == 'Decision Tree':
            classifier = DecisionTreeClassifier(min_samples_split=param)
        elif classifier_name == 'SVM':
            classifier = SVC(coef0=param)
        elif classifier_name == 'KNN':
            classifier = KNeighborsClassifier(leaf_size=param)
        elif classifier_name == 'Logistic Regression':
            classifier = LogisticRegression(max_iter=param)
        elif classifier_name == 'MLP':
            classifier = MLPClassifier(epsilon=param)

        # Perform 3-fold stratified cross-validation
        stratified_kfold = StratifiedKFold(n_splits=3)
        scores = cross_val_score(classifier, X_train, y_train, cv=stratified_kfold)
        st.write(f"Cross-Validation Scores: {scores}")
        st.write(f"Average Cross-Validation Score: {scores.mean():.2f}")

        # Show progress bar for fitting the model
        st.write("Fitting the model to the training data...")
        progress_bar = st.progress(0)
        num_trainings = 100

        for percent_complete in range(num_trainings):
            progress_bar.progress(percent_complete + 1)

        # Fit the model to the training data
        classifier.fit(X_train, y_train)
        st.write("Training completed")

        # Make predictions on the testing data
        y_pred = classifier.predict(X_test)

        # Decode the numeric predictions back to original categorical values (optional)
        y_pred_decoded = label_encoder.inverse_transform(y_pred)
        y_test_decoded = label_encoder.inverse_transform(y_test)

        # Calculate the accuracy score with iteration count
        iteration_count = 100  # Set a fixed number of iterations
        accuracy_values = []

        start_time = time.time()

        for _ in range(iteration_count):
            accuracy_values.append(accuracy_score(y_test, y_pred))

        end_time = time.time()

        average_accuracy = np.mean(accuracy_values)
        elapsed_time = end_time - start_time

        st.write(f"Test Accuracy after {iteration_count} iterations: {average_accuracy:.2f}")
        st.write(f"Time taken for accuracy calculation: {elapsed_time:.2f} seconds")

        # Calculate precision, recall, and F1-score
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        st.write(f"Precision: {precision:.2f}")
        st.write(f"Recall: {recall:.2f}")
        st.write(f"F1-score: {f1:.2f}")

        # Print the classification report with decoded labels
        st.write("Classification Report:")
        st.text(classification_report(y_test_decoded, y_pred_decoded, zero_division='warn'))

        # Compute the confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred, normalize='true')

        # Plot the heatmap for the confusion matrix
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt='.2f', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, cmap='Blues', ax=ax)
        ax.set_title('Confusion Matrix Heatmap')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        st.pyplot(fig)

        # Create a correlation matrix
        corr_matrix = data.corr()

        # Plot the heatmap for the correlation matrix
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=.5, ax=ax)
        ax.set_title('Correlation Matrix Heatmap')
        st.pyplot(fig)

        # Plot the test accuracy over iterations
        fig, ax = plt.subplots()
        ax.plot(range(iteration_count), accuracy_values, label='Test Accuracy per Iteration', marker='o')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Accuracy')
        ax.set_title('Test Accuracy Line Plot')
        ax.legend()
        st.pyplot(fig)
# Sidebar kedua: Penjelasan
elif option == 'Penjelasan':
    #st.sidebar.subheader('Penjelasan')
    st.sidebar.image('gambar.png')
    st.sidebar.write('Ciri-ciri terkena DDoS')
    st.sidebar.write('1. Lalu Lintas Jaringan yang Tidak Biasa: Peningkatan drastis dalam lalu lintas jaringan yang tidak dapat dijelaskan oleh aktivitas pengguna normal. Lalu lintas ini biasanya berasal dari berbagai sumber IP yang berbeda.')
    st.sidebar.write('2. Penurunan Kinerja Server: Server mungkin menjadi sangat lambat atau tidak responsif. Pengguna mungkin mengalami waktu muat halaman yang lama atau tidak dapat mengakses layanan sama sekali.')
    st.sidebar.write('3. Koneksi yang Tidak Biasa: Peningkatan tajam dalam jumlah koneksi aktif, terutama dari alamat IP yang tidak dikenal atau dari berbagai lokasi geografis yang tidak biasa.')

    st.write("Pengelompokkan banyak data :")
    st.write("DDos = 300 dan Normal = 300")
    total_dataset = 600
    jumlah_virus = 300
    jumlah_normal = 300

    persentase_virus = (jumlah_virus / total_dataset) * 100
    persentase_normal = (jumlah_normal / total_dataset) * 100

    st.write(f"Persentase kata virus adalah {persentase_virus:.2f}% dari total dataset.")
    st.write(f"Persentase kata normal adalah {persentase_normal:.2f}% dari total dataset.")

    # Data untuk pie chart
    labels = ['DDoS', 'Normal']
    sizes = [persentase_virus, persentase_normal]
    colors = ['red', 'green']
    explode = (0.1, 0)  # memisahkan irisan pertama (Virus)

    # Membuat pie chart
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    ax.axis('equal')  # Menjaga pie chart berbentuk lingkaran
    ax.set_title('Persentase Virus dan Normal dalam Dataset')
    st.pyplot(fig)

# Sidebar ketiga: Lainnya
elif option == 'Report':
    st.sidebar.subheader('Data Rekap Evaluation Metrics')
    st.sidebar.write('Berdasarkan rekap dari 5 Algoritma :')
    st.sidebar.write('1. Decision Tree')
    st.sidebar.write('2. Support Vector Machine (SVM)')
    st.sidebar.write('3. K-Nearest Neighbors (K-NN)')
    st.sidebar.write('4. Logistic Regression')
    st.sidebar.write('5. Neural Network MLP')

    # Dummy data untuk contoh grafik line
    results = {
    "Decision Tree": {
        "params": [1, 2, 3, 4],
        #"Average Cross-Validation Score": [0.99, 0.50],
        'Test Accuracy': ['50%', '50%', '50%', '50%'],
        'Precision': ['25%', '25%', '25%', '25%'],
        'Recall': ['50%', '50%', '50%', '50%'],
        'F1-score': ['33%', '33%', '33%', '33%']
    },
    "SVM": {
        "params": [1, 2],
        #"Average Cross-Validation Score": [0.99, 0.97],
        'Test Accuracy': ['98%', '98%'],
        'Precision': ['98%', '98%'],
        'Recall': ['98%', '98%'],
        'F1-score': ['98%', '98%']
    },
    "K-NN": {
        "params": [1, 2],
        #"Average Cross-Validation Score": [0.99, 0.99],
        'Test Accuracy': ['99%', '99%'],
        'Precision': ['99%', '99%'],
        'Recall': ['99%', '99%'],
        'F1-score': ['99%', '99%']
    },
    "Logistic Regression": {
        "params": [1, 2, 3, 4],
        #"Average Cross-Validation Score": [0.97, 0.92],
        'Test Accuracy': ['95%', '94%', '93', '93'],
        'Precision': ['95%', '95%', '94%', '95%'],
        'Recall': ['95%', '94%', '93%', '95%'],
        'F1-score': ['95%', '94%', '93%', '95%']
    },
    "Neural Network (MLP)": {
        "params": [1, 2, 3, 4],
        #"Average Cross-Validation Score": [0.98, 0.96],
        'Test Accuracy': ['93%', '97%', '83%', '92%'],
        'Precision': ['93%', '98%', '86%', '93%'],
        'Recall': ['93%', '97%', '83%', '92%'],
        'F1-score': ['92%', '97%', '83%', '92%'],
    }
    }

# Fungsi untuk membuat grafik line gabungan
    def plot_combined_results(results, metric):
        fig, ax = plt.subplots(figsize=(14, 8))
        for algorithm_name, result in results.items():
            ax.plot(result["params"], result[metric], marker='o', label=f"{algorithm_name}")
        ax.set_title(f'{metric} Comparison')
        ax.set_xlabel('Parameter')
        ax.set_ylabel(metric)
        ax.legend()
        ax.grid(True)
        return fig

# Daftar metrik yang akan diplot
    metrics = ["Test Accuracy", "Precision", "Recall", "F1-score"]

# Plot gabungan untuk setiap metrik
    for metric in metrics:
        st.write(f'### {metric} Comparison')
        fig = plot_combined_results(results, metric)
        st.pyplot(fig)


# Sidebar keempat: Lainnya
elif option == 'Pembandingan':
    st.sidebar.subheader('Data Rekap Pembandingan dengan 2 skenario (80:20) dan (70:30)')
    st.sidebar.write('Dengan 5 Algoritma dan masing-masing parameter :')
    st.sidebar.write('1. Decision Tree = min_samples_split dan min_samples_leaf')
    st.sidebar.write('2. Support Vector Machine (SVM) = C dan coef0')
    st.sidebar.write('3. K-Nearest Neighbors (K-NN) = leaf_size dan n_neighbors')
    st.sidebar.write('4. Logistic Regression = tol dan max_iter')
    st.sidebar.write('5. Neural Network MLP = epsilon dan max_iter')

    # Fungsi untuk membuat grafik gabungan
    def plot_combined_metrics(data_80_20, data_70_30, title):
        fig, ax = plt.subplots()
        for metric in ['Test Accuracy', 'Precision', 'Recall', 'F1-Score']:
            ax.plot(data_80_20['Parameters'], data_80_20[metric], marker='o', label=f'{metric} (80:20)')
            ax.plot(data_70_30['Parameters'], data_70_30[metric], marker='x', label=f'{metric} (70:30)')
        ax.set_title(title)
        ax.set_xlabel('Parameter')
        ax.set_ylabel('Value (%)')
        ax.legend()
        st.pyplot(fig)

    # Data untuk setiap classifier
    data_decision_tree_80_20 = {
        'Parameters': ['min_samples_split', 'min_samples_leaf'],
        'Test Accuracy': [50, 50],
        'Precision': [25, 25],
        'Recall': [50, 50],
        'F1-Score': [33, 33]
    }
    data_decision_tree_70_30 = {
        'Parameters': ['min_samples_split', 'min_samples_leaf'],
        'Test Accuracy': [50, 50],
        'Precision': [25, 25],
        'Recall': [50, 50],
        'F1-Score': [33, 33]
    }

    data_svm_80_20 = {
        'Parameters': ['C', 'coef0'],
        'Test Accuracy': [98, 98],
        'Precision': [98, 98],
        'Recall': [98, 98],
        'F1-Score': [98, 98]
    }
    data_svm_70_30 = {
        'Parameters': ['C', 'coef0'],
        'Test Accuracy': [99, 99],
        'Precision': [99, 99],
        'Recall': [99, 99],
        'F1-Score': [99, 99]
    }

    data_knn_80_20 = {
        'Parameters': ['leaf_size', 'n_neighbors'],
        'Test Accuracy': [99, 97],
        'Precision': [99, 98],
        'Recall': [99, 97],
        'F1-Score': [99, 97]
    }
    data_knn_70_30 = {
        'Parameters': ['leaf_size', 'n_neighbors'],
        'Test Accuracy': [99, 98],
        'Precision': [99, 98],
        'Recall': [99, 98],
        'F1-Score': [99, 98]
    }

    data_logreg_80_20 = {
        'Parameters': ['tol (1)', 'tol (2)', 'max_iter (1)', 'max_iter (2)'],
        'Test Accuracy': [95, 94, 93, 95],
        'Precision': [95, 95, 94, 95],
        'Recall': [95, 94, 93, 95],
        'F1-Score': [95, 94, 93, 95]
    }
    data_logreg_70_30 = {
        'Parameters': ['tol (1)', 'tol (2)', 'max_iter (1)', 'max_iter (2)'],
        'Test Accuracy': [96, 94, 94, 94],
        'Precision': [96, 95, 95, 95],
        'Recall': [96, 94, 94, 94],
        'F1-Score': [96, 94, 94, 94]
    }

    data_nn_80_20 = {
        'Parameters': ['epsilon (1)', 'epsilon (2)', 'max_iter (1)', 'max_iter (2)'],
        'Test Accuracy': [93, 97, 83, 92],
        'Precision': [93, 98, 86, 93],
        'Recall': [93, 97, 83, 92],
        'F1-Score': [92, 97, 83, 92]
    }
    data_nn_70_30 = {
        'Parameters': ['epsilon (1)', 'epsilon (2)', 'max_iter (1)', 'max_iter (2)'],
        'Test Accuracy': [90, 97, 83, 97],
        'Precision': [90, 97, 88, 97],
        'Recall': [90, 97, 83, 97],
        'F1-Score': [90, 97, 83, 97]
    }

    # Plot untuk Decision Tree
    st.subheader('1. Decision Tree')
    plot_combined_metrics(data_decision_tree_80_20, data_decision_tree_70_30, 'Decision Tree')

    # Plot untuk SVM
    st.subheader('2. Support Vector Machine')
    plot_combined_metrics(data_svm_80_20, data_svm_70_30, 'Support Vector Machine')

    # Plot untuk K-NN
    st.subheader('3. K-Nearest Neighbor')
    plot_combined_metrics(data_knn_80_20, data_knn_70_30, 'K-Nearest Neighbor')

    # Plot untuk Logistic Regression
    st.subheader('4. Logistic Regression')
    plot_combined_metrics(data_logreg_80_20, data_logreg_70_30, 'Logistic Regression')

    # Plot untuk Neural Network
    st.subheader('5. Neural Network')
    plot_combined_metrics(data_nn_80_20, data_nn_70_30, 'Neural Network')


# Sidebar kelima: Pembandingan
if option == 'Hasil':
    
    st.sidebar.write('Peneliti ingin membandingkan parameter dari setiap algoritma.')
    st.sidebar.write('1. Decision Tree')
    st.sidebar.write('2. Support Vector Machine (SVM)')
    st.sidebar.write('3. K-Nearest Neighbors (K-NN)')
    st.sidebar.write('4. Logistic Regression')
    st.sidebar.write('5. Neural Network MLP')
    
    # Data for comparison
    data_saya1 = {
        'Algoritma Supervised Learning': ['Decision Tree', 'SVM', 'K-NN', 'Logistic Regression', 'Neural Network '],
        'Parameter': ['min_samples_split dan min_samples_leaf', 'C dan coef0', 'leaf_size', 'tol', 'epsilon'],
        'Test Accuracy': ['50%', '98%', '99%', '95%', '97%'],
        'Precision': ['25%', '98%', '99%', '95%', '98%'],
        'Recall': ['50%', '98%', '99%', '95%', '97%'],
        'F1-score': ['33%', '98%', '99%', '95%', '97%'],
    }
    df_saya1 = pd.DataFrame(data_saya1)

    st.subheader("Hasil Terbaik dari Seluruh Model")
    st.dataframe(df_saya1)

    # Convert percentage strings to float
    df_saya1['Test Accuracy'] = df_saya1['Test Accuracy'].str.rstrip('%').astype('float') / 100.0
    df_saya1['Precision'] = df_saya1['Precision'].str.rstrip('%').astype('float') / 100.0
    df_saya1['Recall'] = df_saya1['Recall'].str.rstrip('%').astype('float') / 100.0
    df_saya1['F1-score'] = df_saya1['F1-score'].str.rstrip('%').astype('float') / 100.0

    # Create bar chart for Test Accuracy, Precision, Recall, and F1-score
    fig, ax = plt.subplots(figsize=(12, 8))
    bar_width = 0.2
    index = np.arange(len(df_saya1))

    bar1 = plt.bar(index, df_saya1['Test Accuracy'], bar_width, label='Test Accuracy')
    bar2 = plt.bar(index + bar_width, df_saya1['Precision'], bar_width, label='Precision')
    bar3 = plt.bar(index + 2 * bar_width, df_saya1['Recall'], bar_width, label='Recall')
    bar4 = plt.bar(index + 3 * bar_width, df_saya1['F1-score'], bar_width, label='F1-score')

    plt.xlabel('Models')
    plt.ylabel('Scores')
    plt.title('Performance Metrics for Various Models')
    plt.xticks(index + bar_width * 1.5, df_saya1['Algoritma Supervised Learning'])
    plt.legend()

    st.pyplot(fig)


# Footer dengan warna biru tua
footer_html = """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #003366;
        color: white;
        text-align: center;
        padding: 10px;
    }
    </style>
    <div class="footer">
        Halo saya Muhammad Rafli, mahasiswa tingkat akhir di Politeknik Negeri Semarang
    </div>
    """
st.markdown(footer_html, unsafe_allow_html=True)
