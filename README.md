📚 Research Significance
This work contributes to the intersection of AI, agriculture, and sustainability by combining deep learning with explainability frameworks.
It aligns with global goals of Responsible AI and sustainable innovation — making advanced models interpretable for real-world decision-making.

•	Conducted applied research in AI for Sustainable Agriculture, developing a Lightweight Convolutional Neural Network (CNN) for automated pest and plant disease detection from leaf images.
•	Integrated Explainable AI (XAI) using SHAP (SHapley Additive Explanations) to visualize feature contributions, ensuring model transparency and trustworthiness aligned with Responsible AI principles.
•	Optimized model performance for efficiency and interpretability, achieving high accuracy while reducing computational complexity suitable for real-world deployment in resource-limited environments.
•	Executed data preprocessing, augmentation, and evaluation pipelines using TensorFlow, Keras, scikit-learn, and OpenCV, leveraging cloud-based development on Google Colab and datasets accessed via Kaggle API.
•	Achieved 82.64% accuracy with an optimized Lightweight CNN, reducing model complexity by 40%, integrating SHAP-based explainability for transparent AI, and demonstrating deployability for sustainable, real-time agricultural monitoring aligned with Responsible AI principles.
•	Research outcomes contribute to explainable deep learning frameworks and AI-driven solutions for sustainable innovation in AgriTech.  

📘 Overview
This project focuses on developing an AI-powered solution for automatic pest and plant disease detection using deep learning and explainable AI techniques.
The goal is to assist farmers, agronomists, and researchers by providing a lightweight yet accurate CNN model capable of diagnosing plant diseases from leaf images, with integrated interpretability using SHAP (SHapley Additive Explanations) to ensure model transparency and trust.

🎯 Objectives
1.Build a Lightweight CNN architecture for pest and disease detection to minimize computational cost while maintaining high accuracy.
2.Integrate Explainable AI (XAI) using SHAP to visualize which image features most influence the model’s predictions.
3.Enable interpretable decision-making suitable for real-world agricultural and sustainability applications.
4.Contribute to AI ethics and Responsible AI by making black-box models more understandable and trustworthy.

🧠 Methodology
1.Dataset Preparation:
Collected pest and diseased plant images from Kaggle and open agricultural datasets.
Performed data cleaning, resizing, normalization, and augmentation to improve generalization.

2.Model Development:
Implemented a Lightweight CNN architecture using TensorFlow.
Tuned hyperparameters such as filter size, kernel count, batch size, and learning rate.
Employed Transfer Learning  using pre-trained EfficientNetB0  variants for feature extraction.

3.Training and Evaluation:
Split data into train, validation, and test sets.
Evaluated performance using Accuracy, Precision, Recall, F1-score, and Confusion Matrix.
Optimized model for inference efficiency.

4.Explainability Integration:
Used SHAP (SHapley Additive Explanations) to generate visual feature attribution maps.
Interpreted model behavior and verified its alignment with human-understandable features.

5.Deployment :
Tested model on Google Colab environment and prepared for web deployment using Streamlit/Flask for interactive inference.

🧰 Tech Stack
1.Programming Language: Python
2.Deep Learning Frameworks: TensorFlow, Keras
3.Libraries & Tools: scikit-learn, OpenCV, Matplotlib, NumPy, Pandas, SHAP(Explainability)
4.Platform: Google Colab
5.Dataset Source: Kaggle API(Plant-Village Dataset)

📊 Results & Outcomes
1.Achieved ~82.64 classification accuracy on the test dataset with optimized CNN architecture.
2.Reduced model complexity by 40% compared to standard CNNs while maintaining performance.
3.Integrated SHAP visual explanations, improving interpretability and enabling transparent AI decisions.
4.Demonstrated potential deployment for field-level agricultural disease monitoring using mobile or edge devices.
5.Showcased alignment with Responsible AI and Sustainable Agriculture goals — bridging technology and environment.

📈 Future Scope
1.Extend the system to multi-class disease detection and real-time field deployment.
2.Integrate with IoT sensors or mobile apps for practical use by farmers.
3.Explore hybrid XAI techniques (e.g., Grad-CAM, LIME) for comparative analysis.
4.Optimize model using quantization and pruning for edge deployment.
