# 🚀 GitHub Profile Extractor with ML Analysis

An advanced GitHub profile analyzer that uses machine learning to extract deep technical insights from user repositories. This Streamlit application goes beyond basic profile information to understand developers' technical interests, expertise areas, and project patterns.

## ✨ Features

### 📊 **Basic Profile Analysis**

- User information (name, bio, location, followers, etc.)
- Repository count and activity metrics
- Top 3 most used programming languages

### 🤖 **AI-Powered Technical Analysis**

- **Technical Domain Classification**: Automatically identifies specialization areas (Web Development, Data Science, DevOps, etc.)
- **TF-IDF Analysis**: Uses machine learning to extract key technologies and frameworks
- **Project Type Detection**: Identifies patterns in project types (API Development, Tools & Utilities, etc.)
- **Technical Interest Scoring**: Quantifies technical engagement and expertise breadth (0-100 scale)

### 📈 **Supported Technical Domains**

- Web Development
- Data Science & AI
- Mobile Development
- DevOps & Cloud
- Game Development
- Cybersecurity
- Blockchain
- IoT (Internet of Things)
- Desktop Applications
- Database Management

### 💾 **Export Options**

- Individual JSON files for each user
- Combined technical analysis results
- Local file storage with organized structure

## 🛠️ Installation

### Prerequisites

- Python 3.7+
- pip package manager

### Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/YOUR_USERNAME/github-profile-extractor.git
   cd github-profile-extractor
   ```

2. **Create virtual environment** (recommended)

   ```bash
   python -m venv py_env
   # Windows
   py_env\Scripts\activate
   # Mac/Linux
   source py_env/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install streamlit requests pandas scikit-learn nltk textblob numpy
   ```

## 🚀 Usage

1. **Start the application**

   ```bash
   streamlit run program.py
   ```

2. **Open your browser** and navigate to `http://localhost:8501`

3. **Enter GitHub usernames** (comma-separated for multiple users)

   ```
   Example: octocat, defunkt, torvalds
   ```

4. **View results** including:

   - 🔧 Top Programming Languages
   - 🎯 Technical Domains
   - 🚀 Project Types
   - ⚡ Key Technologies
   - 📊 Technical Interest Score

5. **Download analysis** as JSON for further processing

## 📁 Project Structure

```
github-profile-extractor/
├── program.py                 # Main Streamlit application
├── .gitignore                # Git ignore rules
├── README.md                 # This file
└── github_data_exports/      # Generated data (not tracked in git)
    ├── user1_github_data.json
    └── user2_github_data.json
```

## 🔬 How the ML Analysis Works

### **1. Data Collection**

- Fetches repository names, descriptions, and topics from GitHub API
- Combines textual data for comprehensive analysis

### **2. Domain Classification**

- Uses keyword matching against predefined technical domains
- Scores domains based on keyword frequency and relevance

### **3. TF-IDF Analysis**

```python
vectorizer = TfidfVectorizer(
    max_features=50,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=1
)
```

- **Term Frequency-Inverse Document Frequency** identifies important technical terms
- Discovers frameworks and technologies specific to each developer

### **4. Pattern Recognition**

- Analyzes repository patterns to identify project types
- Uses boolean logic and text matching for classification

### **5. Interest Scoring**

```python
interest_score = len(repos) * 10 + len(languages) * 5 + len(domains) * 3
```

- Combines repository activity, language diversity, and domain breadth
- Provides quantified technical engagement metric

## 📊 Example Output

```json
{
  "username": "octocat",
  "name": "The Octocat",
  "primary_languages": ["JavaScript", "Ruby", "Python"],
  "technical_domains": ["Web Development", "DevOps"],
  "project_types": ["API Development", "Tools & Utilities"],
  "key_technologies": ["react", "docker", "nodejs", "rails"],
  "technical_interest_score": 85
}
```

## 🔧 Configuration

### **Customizing Technical Domains**

Edit the `technical_keywords` dictionary in `extract_technical_interests_ml()` to add new domains:

```python
technical_keywords = {
    "Your New Domain": ["keyword1", "keyword2", "keyword3"],
    # ... existing domains
}
```

### **Adjusting TF-IDF Parameters**

Modify the TfidfVectorizer parameters for different analysis sensitivity:

```python
vectorizer = TfidfVectorizer(
    max_features=100,     # Increase for more detailed analysis
    ngram_range=(1, 3),   # Include 3-word phrases
    min_df=2              # Require terms to appear in multiple repos
)
```

## 🚫 Rate Limiting

- GitHub API allows 60 requests/hour for unauthenticated requests
- For higher limits, add GitHub token authentication
- The app processes multiple users sequentially

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m "Add amazing feature"`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **Streamlit** for the amazing web app framework
- **scikit-learn** for machine learning capabilities
- **GitHub API** for providing comprehensive developer data
- **TF-IDF algorithm** for intelligent text analysis

## 📧 Contact

For questions, suggestions, or issues, please open an issue on GitHub or contact the maintainer.

---

**Built with ❤️ using Python, Streamlit, and Machine Learning**
