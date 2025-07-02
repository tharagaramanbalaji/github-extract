import streamlit as st
import requests
import json
import os
import pandas as pd
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from textblob import TextBlob
import numpy as np

def get_github_user(username):
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

def get_user_repos(username):
    url = f"https://api.github.com/users/{username}/repos"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return []

def get_user_events(username):
    url = f"https://api.github.com/users/{username}/events/public"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return []

def extract_top_languages(repos):
    """Extract top 3 programming languages from user repositories"""
    languages = []
    for repo in repos:
        if repo.get('language') and repo.get('language') != 'null':
            languages.append(repo['language'])
    
    if not languages:
        return []
    
    # Count occurrences of each language
    language_counts = Counter(languages)
    # Get top 3 most common languages
    top_languages = [lang for lang, count in language_counts.most_common(3)]
    
    return top_languages

def extract_technical_interests_ml(repos):
    """Use ML to extract technical interests from repository data"""
    if not repos:
        return {
            "primary_languages": [],
            "technical_domains": [],
            "project_types": [],
            "frameworks_tech": [],
            "interest_score": 0
        }
    
    # Collect text data from repositories
    repo_texts = []
    languages = []
    topics_keywords = []
    
    for repo in repos:
        # Collect repository descriptions, names, and topics
        text_data = []
        if repo.get('name'):
            text_data.append(repo['name'])
        if repo.get('description'):
            text_data.append(repo['description'])
        if repo.get('topics'):
            text_data.extend(repo['topics'])
            topics_keywords.extend(repo['topics'])
        
        # Collect languages
        if repo.get('language') and repo.get('language') != 'null':
            languages.append(repo['language'])
        
        repo_text = ' '.join(text_data)
        if repo_text.strip():
            repo_texts.append(repo_text)
    
    # Analyze programming languages
    language_counts = Counter(languages)
    primary_languages = [lang for lang, count in language_counts.most_common(5)]
    
    # Extract technical domains using keyword analysis
    technical_keywords = {
        'Web Development': ['web', 'frontend', 'backend', 'fullstack', 'website', 'http', 'api', 'rest', 'graphql'],
        'Data Science': ['data', 'analytics', 'machine learning', 'ml', 'ai', 'artificial intelligence', 'dataset', 'analysis'],
        'Mobile Development': ['mobile', 'android', 'ios', 'react native', 'flutter', 'swift', 'kotlin'],
        'DevOps': ['docker', 'kubernetes', 'ci/cd', 'deployment', 'infrastructure', 'cloud', 'aws', 'azure'],
        'Game Development': ['game', 'unity', 'unreal', 'gaming', 'pygame', 'gamedev'],
        'Cybersecurity': ['security', 'encryption', 'hacking', 'cybersecurity', 'penetration', 'vulnerability'],
        'Blockchain': ['blockchain', 'cryptocurrency', 'bitcoin', 'ethereum', 'smart contract', 'defi'],
        'IoT': ['iot', 'internet of things', 'sensor', 'arduino', 'raspberry pi', 'embedded'],
        'Desktop Applications': ['desktop', 'gui', 'tkinter', 'qt', 'electron', 'wpf'],
        'Database': ['database', 'sql', 'mongodb', 'postgresql', 'mysql', 'nosql']
    }
    
    # Combine all text for analysis
    all_text = ' '.join(repo_texts + topics_keywords).lower()
    
    # Identify technical domains
    identified_domains = []
    for domain, keywords in technical_keywords.items():
        score = sum(1 for keyword in keywords if keyword in all_text)
        if score > 0:
            identified_domains.append((domain, score))
    
    # Sort by relevance score
    identified_domains.sort(key=lambda x: x[1], reverse=True)
    technical_domains = [domain for domain, score in identified_domains[:3]]
    
    # Extract frameworks and technologies using TF-IDF if we have enough text
    frameworks_tech = []
    if repo_texts and len(repo_texts) >= 2:
        try:
            # Use TF-IDF to find important technical terms
            vectorizer = TfidfVectorizer(
                max_features=50,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1
            )
            tfidf_matrix = vectorizer.fit_transform(repo_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get average TF-IDF scores
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            
            # Get top technical terms
            top_indices = mean_scores.argsort()[-10:][::-1]
            frameworks_tech = [feature_names[i] for i in top_indices if len(feature_names[i]) > 2][:5]
        except:
            # Fallback to simple keyword extraction
            frameworks_tech = list(set(topics_keywords))[:5]
    
    # Determine project types based on repository patterns
    project_types = []
    if any('api' in text.lower() for text in repo_texts):
        project_types.append('API Development')
    if any('bot' in text.lower() for text in repo_texts):
        project_types.append('Bot Development')
    if any('tool' in text.lower() or 'utility' in text.lower() for text in repo_texts):
        project_types.append('Tools & Utilities')
    if any('library' in text.lower() or 'package' in text.lower() for text in repo_texts):
        project_types.append('Libraries & Packages')
    
    # Calculate overall interest score based on repository activity
    interest_score = len(repos) * 10 + len(set(languages)) * 5 + len(technical_domains) * 3
    
    return {
        "primary_languages": primary_languages,
        "technical_domains": technical_domains,
        "project_types": project_types,
        "frameworks_tech": frameworks_tech,
        "interest_score": min(interest_score, 100)  # Cap at 100
    }


st.title("Advanced GitHub Profile Extractor (Multiple Users)")

usernames = st.text_area("Enter one or more GitHub usernames (comma-separated):", "")
if usernames:
    username_list = [name.strip() for name in usernames.split(",") if name.strip()]
    results = []
    errors = []
    save_dir = "github_data_exports"
    os.makedirs(save_dir, exist_ok=True)

    for username in username_list:
        user_data = get_github_user(username)
        if user_data:
            repos = get_user_repos(username)
            events = get_user_events(username)
            
            # Extract top 3 technical preferences (basic)
            top_languages = extract_top_languages(repos)
            
            # Extract advanced technical interests using ML
            ml_analysis = extract_technical_interests_ml(repos)

            combined_data = {
                "user": user_data,
                "repos": repos,
                "recent_events": events[:5],
                "top_technical_preferences": top_languages,
                "ml_technical_analysis": ml_analysis
            }
            results.append(combined_data)

            # Save individual JSON
            file_path = os.path.join(save_dir, f"{username}_github_data.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(combined_data, f, indent=4)

        else:
            errors.append(username)

    # Display combined results
    if results:
        st.subheader("✅ Retrieved User Profiles")
        
        # Create CSV data
        csv_data = []
        for result in results:
            user = result["user"]
            top_langs = result["top_technical_preferences"]
            ml_analysis = result["ml_technical_analysis"]
            
            # Display each user with their enhanced analysis
            st.write(f"**{user['name'] or user['login']} (@{user['login']})**")
            
            # Basic language preferences
            if top_langs:
                st.write(f"🔧 **Top Programming Languages:** {', '.join(top_langs)}")
            else:
                st.write("🔧 **Top Programming Languages:** No languages detected")
            
            # ML-based technical analysis
            st.write(f"🎯 **Technical Domains:** {', '.join(ml_analysis['technical_domains'][:3]) if ml_analysis['technical_domains'] else 'Not identified'}")
            st.write(f"🚀 **Project Types:** {', '.join(ml_analysis['project_types'][:3]) if ml_analysis['project_types'] else 'Not identified'}")
            st.write(f"⚡ **Key Technologies:** {', '.join(ml_analysis['frameworks_tech'][:3]) if ml_analysis['frameworks_tech'] else 'Not identified'}")
            st.write(f"📊 **Technical Interest Score:** {ml_analysis['interest_score']}/100")
            st.write("---")
            
            # Prepare enhanced CSV row
            csv_row = {
                "Name": user.get('name', ''),
                "Username": user.get('login', ''),
                "Company": user.get('company', ''),
                "Location": user.get('location', ''),
                "Bio": user.get('bio', ''),
                "Public Repos": user.get('public_repos', 0),
                "Followers": user.get('followers', 0),
                "Following": user.get('following', 0),
                "Primary Language 1": top_langs[0] if len(top_langs) > 0 else '',
                "Primary Language 2": top_langs[1] if len(top_langs) > 1 else '',
                "Primary Language 3": top_langs[2] if len(top_langs) > 2 else '',
                "Technical Domain 1": ml_analysis['technical_domains'][0] if len(ml_analysis['technical_domains']) > 0 else '',
                "Technical Domain 2": ml_analysis['technical_domains'][1] if len(ml_analysis['technical_domains']) > 1 else '',
                "Technical Domain 3": ml_analysis['technical_domains'][2] if len(ml_analysis['technical_domains']) > 2 else '',
                "Project Type 1": ml_analysis['project_types'][0] if len(ml_analysis['project_types']) > 0 else '',
                "Project Type 2": ml_analysis['project_types'][1] if len(ml_analysis['project_types']) > 1 else '',
                "Key Technology 1": ml_analysis['frameworks_tech'][0] if len(ml_analysis['frameworks_tech']) > 0 else '',
                "Key Technology 2": ml_analysis['frameworks_tech'][1] if len(ml_analysis['frameworks_tech']) > 1 else '',
                "Key Technology 3": ml_analysis['frameworks_tech'][2] if len(ml_analysis['frameworks_tech']) > 2 else '',
                "Technical Interest Score": ml_analysis['interest_score'],
                "Profile URL": user.get('html_url', '')
            }
            csv_data.append(csv_row)
        
        # Create DataFrame and CSV
        df = pd.DataFrame(csv_data)
        csv_string = df.to_csv(index=False)
        
        st.json(results)

        # Download button for ML analysis results only
        analysis_results = []
        for result in results:
            user = result["user"]
            ml_analysis = result["ml_technical_analysis"]
            
            analysis_data = {
                "username": user.get('login', ''),
                "name": user.get('name', ''),
                "primary_languages": ml_analysis['primary_languages'],
                "technical_domains": ml_analysis['technical_domains'],
                "project_types": ml_analysis['project_types'],
                "key_technologies": ml_analysis['frameworks_tech'],
                "technical_interest_score": ml_analysis['interest_score']
            }
            analysis_results.append(analysis_data)
        
        analysis_json = json.dumps(analysis_results, indent=4)
        st.download_button(
            label="Download Technical Analysis Results as JSON",
            data=analysis_json,
            file_name="github_technical_analysis.json",
            mime="application/json"
        )
        
        st.success(f"Individual files saved to `{save_dir}` directory.")

    # Display errors
    if errors:
        st.error(f"❌ Could not retrieve data for: {', '.join(errors)}")
