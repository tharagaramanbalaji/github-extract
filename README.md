# 🚀 GitHub Profile Extractor with ML Analysis

An advanced GitHub profile analyzer that uses machine learning to extract deep technical insights from user repositories. This Streamlit application goes beyond basic profile information to understand developers' technical interests, expertise areas, and project patterns.

## ✨ Features

### 📊 **Basic Profile Analysis**

- User information (name, bio, location, followers, etc.)
- Repository count and activity metrics
- Top 3 most used programming languages
- Flexible username input (URLs, comma-separated, space-separated, or newline-separated)
- Spelling mistake handling with fuzzy matching

### 🤖 **AI-Powered Technical Analysis**

- **Technical Domain Classification**: Automatically identifies specialization areas
- **TF-IDF Analysis**: Uses machine learning to extract key technologies and frameworks
- **Project Type Detection**: Identifies patterns in project types
- **Technical Interest Scoring**: Quantifies technical engagement (0-100 scale)
- **Domain Expertise Validation**: Verifies claimed expertise against actual work

### 🎯 **Domain Expertise Analysis**

- **Claimed vs. Verified Domains**: Compares bio claims with repository evidence
- **Proficiency Scoring**: Detailed scoring for each technical domain
- **Expertise Levels**:
  - 🟢 Expert (70%+ consistency)
  - 🟡 Intermediate (40-70% consistency)
  - 🔴 Beginner (<40% consistency)
- **Enhancement Suggestions**: AI-powered recommendations for profile improvement

### 📈 **Supported Technical Domains**

- AI/ML & Data Science
- Web Development
- Mobile Development
- DevOps & Cloud
- Cybersecurity
- Game Development
- Embedded/IoT
- And more...

### 💾 **Export Options**

- Individual JSON files for each user
- Combined technical analysis results
- Formatted CSV exports
- Local file storage with organized structure

## 🔬 How the Analysis Works

### **1. Domain Expertise Validation**

```python
# Three-factor scoring for each domain
repo_score = matching_repos / total_repos * 100       # 40% weight
lang_score = matching_languages / expected * 100       # 30% weight
domain_match = ML_analysis_confirmation * 100         # 30% weight

final_score = (repo_score * 0.4 + lang_score * 0.3 + domain_match * 0.3)
```

### **2. Profile Consistency Check**

- Analyzes bio claims against repository evidence
- Validates programming languages and frameworks
- Checks project types and technical domains
- Generates improvement suggestions

### **3. Technical Interest Score**

```python
interest_score = (
    (repo_count / 50) * 30 +           # Repository activity
    (language_diversity / 10) * 15 +    # Language diversity
    (domain_breadth / 5) * 15 +        # Domain breadth
    (avg_stars / 100) * 15 +           # Project impact
    (recent_activity / 20) * 15 +      # Recent engagement
    (follower_count / 500) * 10        # Community influence
)
```

## 📊 Example Output

```json
{
  "username": "developer123",
  "name": "Jane Developer",
  "domain_expertise": {
    "claimed_domains": ["AI/ML", "Web Development"],
    "verified_domains": ["Web Development", "DevOps"],
    "consistency_scores": {
      "AI/ML": 35.5,
      "Web Development": 85.2,
      "DevOps": 72.1
    },
    "suggestions": [
      "Consider adding more AI/ML projects to support expertise claim",
      "Strong DevOps work could be highlighted in bio"
    ]
  },
  "technical_interest_score": 78.5
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
