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
import urllib.parse
import difflib
from datetime import datetime, timedelta

# Cache for previously successful usernames
successful_usernames = set()

def get_github_user(username):
    url = f"https://api.github.com/users/{username}"
    response = requests.get(url)
    if response.status_code == 200:
        successful_usernames.add(username)  # Add to cache if found
        return response.json()
    return None

def correct_username(username, known_usernames):
    # Suggest closest match if username is not found
    matches = difflib.get_close_matches(username, known_usernames, n=1, cutoff=0.8)
    return matches[0] if matches else username

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

def extract_technical_interests_ml(repos, user_data=None):
    """Use ML to extract technical interests from repository data and user profile"""
    if not repos:
        return {
            "primary_languages": [],
            "technical_domains": [],
            "project_types": [],
            "frameworks_tech": [],
            "interest_score": 0
        }
    
    repo_texts = []
    languages = []
    topics_keywords = []
    repo_stars = []
    recent_activity = 0

    for repo in repos:
        text_data = []
        if repo.get('name'):
            text_data.append(repo['name'])
        if repo.get('description'):
            text_data.append(repo['description'])
        if repo.get('topics'):
            text_data.extend(repo['topics'])
            topics_keywords.extend(repo['topics'])
        if repo.get('language') and repo.get('language') != 'null':
            languages.append(repo['language'])
        if repo.get('stargazers_count'):
            repo_stars.append(repo['stargazers_count'])
        repo_text = ' '.join(text_data)
        if repo_text.strip():
            repo_texts.append(repo_text)
        # Count recent activity (last 6 months)
        if repo.get('updated_at'):
            try:
                updated = datetime.strptime(repo['updated_at'], "%Y-%m-%dT%H:%M:%SZ")
                if updated > datetime.utcnow() - timedelta(days=180):
                    recent_activity += 1
            except:
                pass

    language_counts = Counter(languages)
    primary_languages = [lang for lang, count in language_counts.most_common(5)]

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

    all_text = ' '.join(repo_texts + topics_keywords).lower()
    identified_domains = []
    for domain, keywords in technical_keywords.items():
        score = sum(1 for keyword in keywords if keyword in all_text)
        if score > 0:
            identified_domains.append((domain, score))
    identified_domains.sort(key=lambda x: x[1], reverse=True)
    technical_domains = [domain for domain, score in identified_domains[:3]]

    frameworks_tech = []
    if repo_texts and len(repo_texts) >= 2:
        try:
            vectorizer = TfidfVectorizer(
                max_features=50,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1
            )
            tfidf_matrix = vectorizer.fit_transform(repo_texts)
            feature_names = vectorizer.get_feature_names_out()
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            top_indices = mean_scores.argsort()[-10:][::-1]
            frameworks_tech = [feature_names[i] for i in top_indices if len(feature_names[i]) > 2][:5]
        except:
            frameworks_tech = list(set(topics_keywords))[:5]

    project_types = []
    if any('api' in text.lower() for text in repo_texts):
        project_types.append('API Development')
    if any('bot' in text.lower() for text in repo_texts):
        project_types.append('Bot Development')
    if any('tool' in text.lower() or 'utility' in text.lower() for text in repo_texts):
        project_types.append('Tools & Utilities')
    if any('library' in text.lower() or 'package' in text.lower() for text in repo_texts):
        project_types.append('Libraries & Packages')

    # --- Improved Scoring ---
    # Normalize each factor based on realistic upper bounds
    repo_score = min(len(repos) / 50, 1) * 30  # 50+ repos is exceptional
    lang_score = min(len(set(languages)) / 10, 1) * 15  # 10+ languages is diverse
    domain_score = min(len(technical_domains) / 5, 1) * 15  # 5+ domains is rare
    star_score = min(np.mean(repo_stars) / 100, 1) * 15 if repo_stars else 0  # avg 100+ stars is impressive
    activity_score = min(recent_activity / 20, 1) * 15  # 20+ active repos in 6 months is high
    follower_score = 0
    if user_data and user_data.get('followers') is not None:
        follower_score = min(user_data['followers'] / 500, 1) * 10  # 500+ followers is rare

    interest_score = repo_score + lang_score + domain_score + star_score + activity_score + follower_score
    interest_score = round(min(interest_score, 100), 2)  # Cap at 100, round for clarity

    return {
        "primary_languages": primary_languages,
        "technical_domains": technical_domains,
        "project_types": project_types,
        "frameworks_tech": frameworks_tech,
        "interest_score": interest_score
    }

def validate_domain_consistency(bio: str, repos: list, ml_analysis: dict) -> dict:
    """
    Validate consistency between claimed domains in bio and actual repository content
    """
    # Define domain keywords and their related terms
    domain_expertise = {
        "AI/ML": {
            "bio_keywords": ['ai', 'ml', 'machine learning', 'deep learning', 'artificial intelligence', 'data scientist'],
            "repo_keywords": ['tensorflow', 'pytorch', 'keras', 'scikit-learn', 'neural', 'classifier', 'prediction'],
            "related_languages": ['Python', 'R', 'Julia']
        },
        "Web Development": {
            "bio_keywords": ['web dev', 'frontend', 'backend', 'full stack', 'web developer'],
            "repo_keywords": ['react', 'vue', 'angular', 'node', 'django', 'flask', 'express'],
            "related_languages": ['JavaScript', 'TypeScript', 'HTML', 'CSS', 'PHP']
        },
        "Mobile Development": {
            "bio_keywords": ['mobile dev', 'ios dev', 'android dev', 'app developer'],
            "repo_keywords": ['android', 'ios', 'swift', 'kotlin', 'flutter', 'react-native'],
            "related_languages": ['Java', 'Kotlin', 'Swift', 'Dart']
        },
        "DevOps": {
            "bio_keywords": ['devops', 'sre', 'platform engineer', 'cloud engineer'],
            "repo_keywords": ['docker', 'kubernetes', 'terraform', 'ansible', 'jenkins', 'aws', 'azure'],
            "related_languages": ['Python', 'Shell', 'Go', 'YAML']
        },
        "Cybersecurity": {
            "bio_keywords": ['security', 'pentester', 'cyber', 'infosec', 'security engineer'],
            "repo_keywords": ['security', 'encryption', 'vulnerability', 'exploit', 'firewall', 'penetration'],
            "related_languages": ['Python', 'C', 'Assembly', 'Shell']
        },
        "Game Development": {
            "bio_keywords": ['game dev', 'game developer', 'unity dev', 'unreal dev'],
            "repo_keywords": ['unity', 'unreal', 'godot', 'gamedev', 'pygame', 'directx'],
            "related_languages": ['C#', 'C++', 'Python', 'Lua']
        },
        "Embedded/IoT": {
            "bio_keywords": ['embedded', 'iot', 'firmware', 'hardware'],
            "repo_keywords": ['arduino', 'raspberry pi', 'esp32', 'embedded', 'sensor', 'mqtt'],
            "related_languages": ['C', 'C++', 'Python', 'Assembly']
        }
    }

    bio = bio.lower() if bio else ""
    consistency_report = {
        "claimed_domains": [],
        "verified_domains": [],
        "consistency_scores": {},
        "suggestions": []
    }

    # Check bio for claimed domains
    for domain, keywords in domain_expertise.items():
        if any(keyword in bio for keyword in keywords["bio_keywords"]):
            consistency_report["claimed_domains"].append(domain)

    # Analyze repositories for each domain
    for domain, keywords in domain_expertise.items():
        domain_score = 0
        total_checks = 3  # We'll check repos, languages, and ML analysis
        
        # Check repository keywords
        repo_matches = 0
        for repo in repos:
            repo_text = f"{repo.get('name', '')} {repo.get('description', '')} {' '.join(repo.get('topics', []))}"
            repo_text = repo_text.lower()
            if any(keyword in repo_text for keyword in keywords["repo_keywords"]):
                repo_matches += 1
        repo_score = min(repo_matches / max(len(repos), 1) * 100, 100) if repos else 0
        
        # Check programming languages
        lang_matches = sum(1 for lang in ml_analysis["primary_languages"] 
                         if lang in keywords["related_languages"])
        lang_score = (lang_matches / len(keywords["related_languages"])) * 100
        
        # Check ML-identified domains
        domain_match_score = 100 if domain in ml_analysis["technical_domains"] else 0
        
        # Calculate overall domain consistency
        domain_score = (repo_score + lang_score + domain_match_score) / 3
        consistency_report["consistency_scores"][domain] = round(domain_score, 2)
        
        if domain_score >= 60:
            consistency_report["verified_domains"].append(domain)
        
        # Generate suggestions
        if domain in consistency_report["claimed_domains"] and domain_score < 40:
            consistency_report["suggestions"].append(
                f"Claims {domain} expertise but has limited supporting evidence in repositories"
            )
        elif domain not in consistency_report["claimed_domains"] and domain_score > 70:
            consistency_report["suggestions"].append(
                f"Shows strong {domain} work but doesn't mention it in bio"
            )

    return consistency_report



def parse_usernames(raw_input):
    # Split by comma, whitespace, or newline
    entries = re.split(r'[,\s\n]+', raw_input)
    usernames = set()
    for entry in entries:
        entry = entry.strip()
        if not entry:
            continue
        # If it's a GitHub URL, extract the username
        if entry.startswith("https://github.com/"):
            path = urllib.parse.urlparse(entry).path
            username = path.strip("/").split("/")[0]
            if username:
                usernames.add(username)
        else:
            usernames.add(entry)
    return list(usernames)

def validate_username(username):
    """Validate GitHub username format"""
    # GitHub username rules: 1-39 chars, alphanumeric with single hyphens
    pattern = r'^[a-zA-Z0-9](?:[a-zA-Z0-9]|-(?=[a-zA-Z0-9])){0,38}$'
    return bool(re.match(pattern, username))

def validate_input(raw_input):
    """Validate the input string"""
    if not raw_input or raw_input.isspace():
        return False, "Input cannot be empty"
    if len(raw_input) > 1000:  # Arbitrary limit
        return False, "Input too long"
    return True, ""

def check_rate_limit():
    """Check GitHub API rate limit status"""
    response = requests.get('https://api.github.com/rate_limit')
    if response.status_code == 200:
        data = response.json()
        remaining = data['resources']['core']['remaining']
        reset_time = datetime.fromtimestamp(data['resources']['core']['reset'])
        return remaining, reset_time
    return None, None

def validate_rate_limit():
    remaining, reset_time = check_rate_limit()
    if remaining is not None:
        if remaining < 10:  # Arbitrary threshold
            wait_time = (reset_time - datetime.now()).total_seconds()
            return False, f"Rate limit low ({remaining} remaining). Reset in {int(wait_time/60)} minutes."
    return True, ""

def format_error_message(errors):
    """Format error messages for display"""
    if not errors:
        return ""
    
    error_types = {
        'not_found': [],
        'rate_limit': [],
        'invalid_format': [],
        'other': []
    }
    
    for error in errors:
        if 'not found' in error.lower():
            error_types['not_found'].append(error)
        elif 'rate limit' in error.lower():
            error_types['rate_limit'].append(error)
        elif 'invalid format' in error.lower():
            error_types['invalid_format'].append(error)
        else:
            error_types['other'].append(error)
    
    error_msg = []
    if error_types['not_found']:
        error_msg.append("🔍 Not Found: " + ", ".join(error_types['not_found']))
    if error_types['rate_limit']:
        error_msg.append("⏳ Rate Limited: " + ", ".join(error_types['rate_limit']))
    if error_types['invalid_format']:
        error_msg.append("❌ Invalid Format: " + ", ".join(error_types['invalid_format']))
    if error_types['other']:
        error_msg.append("⚠️ Other Errors: " + ", ".join(error_types['other']))
    
    return "\n".join(error_msg)

st.title("Advanced GitHub Profile Extractor (Multiple Users)")

usernames = st.text_area("Enter one or more GitHub usernames (comma, space, newline, or URL):", "")
if usernames:
    # Validate input first
    is_valid, error_msg = validate_input(usernames)
    if not is_valid:
        st.error(error_msg)
        st.stop()

    # Parse usernames into a list
    username_list = parse_usernames(usernames)
    
    # Initialize results and errors lists
    results = []
    errors = []
    save_dir = "github_data_exports"
    os.makedirs(save_dir, exist_ok=True)

    # Process usernames and collect results
    for username in username_list:
        user_data = get_github_user(username)
        # If not found, try fuzzy correction
        if not user_data and successful_usernames:
            corrected = correct_username(username, successful_usernames)
            if corrected != username:
                st.warning(f"Did you mean `{corrected}` instead of `{username}`?")
                user_data = get_github_user(corrected)
                username = corrected
        if user_data:
            repos = get_user_repos(username)
            events = get_user_events(username)
            
            # Extract top 3 technical preferences (basic)
            top_languages = extract_top_languages(repos)
            
            # Extract advanced technical interests using ML
            ml_analysis = extract_technical_interests_ml(repos, user_data)

            # Get consistency report
            consistency_report = validate_domain_consistency(user_data.get('bio', ''), repos, ml_analysis)
            
            # Add consistency report to the combined data
            combined_data = {
                "user": user_data,
                "repos": repos,
                "recent_events": events[:5],
                "top_technical_preferences": top_languages,
                "ml_technical_analysis": ml_analysis,
                "domain_consistency": consistency_report
            }
            results.append(combined_data)

            # Save individual JSON
            file_path = os.path.join(save_dir, f"{username}_github_data.json")
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(combined_data, f, indent=4)

        else:
            errors.append(username)

    # Display results (keep only this one display section)
    if results:
        st.subheader("✅ Retrieved User Profiles")
        
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
            
            # Domain Expertise Analysis
            with st.expander("🎓 Domain Expertise Analysis", expanded=True):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📋 Claimed Expertise")
                    if consistency_report['claimed_domains']:
                        for domain in consistency_report['claimed_domains']:
                            score = consistency_report['consistency_scores'].get(domain, 0)
                            emoji = "🟢" if score >= 70 else "🟡" if score >= 40 else "🔴"
                            st.markdown(f"{emoji} **{domain}** ({score:.1f}% validated)")
                    else:
                        st.markdown("*No specific expertise claimed in bio*")

                with col2:
                    st.markdown("### 🔍 Discovered Expertise")
                    discovered = [d for d in consistency_report['verified_domains'] 
                                if d not in consistency_report['claimed_domains']]
                    if discovered:
                        for domain in discovered:
                            score = consistency_report['consistency_scores'].get(domain, 0)
                            st.markdown(f"⭐ **{domain}** ({score:.1f}% confidence)")
                    else:
                        st.markdown("*No additional domains discovered*")

                # Detailed Score Breakdown
                st.markdown("### 📊 Domain Proficiency Scores")
                scores_df = pd.DataFrame([
                    {
                        "Domain": domain,
                        "Score": score,
                        "Status": "🟢 Expert" if score >= 70 else "🟡 Intermediate" if score >= 40 else "🔴 Beginner"
                    }
                    for domain, score in consistency_report['consistency_scores'].items()
                    if score > 0  # Only show domains with non-zero scores
                ]).sort_values(by='Score', ascending=False)
                
                st.dataframe(
                    scores_df,
                    column_config={
                        "Domain": "Technical Domain",
                        "Score": st.column_config.NumberColumn(
                            "Proficiency Score",
                            format="%.1f%%"
                        ),
                        "Status": "Expertise Level"
                    },
                    hide_index=True
                )

                # Improvement Suggestions
                if consistency_report["suggestions"]:
                    st.markdown("### 💡 Profile Enhancement Suggestions")
                    for suggestion in consistency_report["suggestions"]:
                        st.markdown(f"- {suggestion}")

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
        st.error(format_error_message(errors))


