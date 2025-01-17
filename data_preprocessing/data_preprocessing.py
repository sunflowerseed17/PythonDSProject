# Importing dependencies
import os
import re
import time
from datetime import datetime, timedelta

import praw
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

# Configure Reddit API
reddit = praw.Reddit(
    client_id="ZaUY5qF9eLVVpD2OvHGEhg",
    client_secret="djHnirfkPnZUNI7XNs4dKUflOKjmtQ",
    user_agent="TextScraper by u/Jammberg",
    check_for_async=False
)

# List of related subreddits for specific scraping
subreddits = ["depression", "breastcancer"]

# Regex pattern for phrases indicating "I have been diagnosed with"
search_pattern = re.compile(
    r"(i\s+(was|am|have been|got|recently got|just got|was just|had been|found out i\s+was|"
    r"was diagnosed as having|diagnosed as suffering from|got diagnosed as having|received a diagnosis of|"
    r"was told i\s+have|was informed i\s+have)\s+.*)",
    re.IGNORECASE
)

# Utility functions

def create_folder(folder_name):
    """Create a folder if it doesn't exist."""
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def save_post(post, output_folder, filename_prefix="post"):
    """Save a Reddit post to a file."""
    filename = f"{filename_prefix}_{post.id}.txt"
    filepath = os.path.join(output_folder, filename)
    try:
        with open(filepath, "w", encoding="utf-8") as file:
            file.write(f"Subreddit: {post.subreddit.display_name}\n")
            file.write(f"Title: {post.title}\n")
            file.write(f"Author: {post.author}\n")
            file.write(f"Score: {post.score}\n")
            file.write(f"Created UTC: {datetime.utcfromtimestamp(post.created_utc)}\n")
            file.write(f"URL: {post.url}\n")
            file.write("\n")
            file.write(post.selftext)
        print(f"Saved post to {filepath}")
    except Exception as e:
        print(f"Error saving post {post.id}: {e}")

def fetch_user_posts(username, reference_date, output_folder, max_posts=None):
    """Fetch posts from a specific user."""
    if not username:
        print("Author not available for this post.")
        return 0

    post_count = 0
    try:
        user = reddit.redditor(username)
        one_month_ago = reference_date - timedelta(days=30)
        for post in user.submissions.new(limit=None):
            if max_posts and post_count >= max_posts:
                break
            post_date = datetime.utcfromtimestamp(post.created_utc)
            if one_month_ago <= post_date <= reference_date and post.selftext.strip():
                save_post(post, output_folder, filename_prefix="user")
                post_count += 1
            elif post_date < one_month_ago:
                break
            time.sleep(2)  # Avoid hitting rate limits
    except Exception as e:
        print(f"Error fetching posts for user {username}: {e}")

    return post_count

def fetch_posts_from_subreddit(subreddit_name):
    """Fetch posts from a specific subreddit."""
    print(f"\nFetching posts from r/{subreddit_name}...\n")
    subreddit = reddit.subreddit(subreddit_name)
    output_folder = f"data/reddit_scraped_posts/{subreddit_name}"
    create_folder(output_folder)

    try:
        for post in subreddit.new(limit=None):
            if post.selftext.strip() and re.search(search_pattern, post.selftext):
                reference_date = datetime.utcfromtimestamp(post.created_utc)
                save_post(post, output_folder, filename_prefix="subreddit")
                fetch_user_posts(post.author.name, reference_date, output_folder)
            time.sleep(2)  # Avoid hitting rate limits
    except Exception as e:
        print(f"Error fetching posts from r/{subreddit_name}: {e}")

def fetch_posts_from_all(max_posts=1100, max_retries=3):
    """Fetch posts from r/all (new) within the last 2 months."""
    print("\nFetching posts from r/all (new)...\n")
    subreddit = reddit.subreddit("all")
    output_folder = "data/reddit_scraped_posts/standard"
    create_folder(output_folder)

    cutoff_date = datetime.utcnow() - timedelta(days=60)
    post_count = 0
    attempts = 0

    while attempts < max_retries:
        try:
            for post in subreddit.new(limit=None):
                if post_count >= max_posts:
                    break
                post_date = datetime.utcfromtimestamp(post.created_utc)
                if post_date < cutoff_date:
                    break
                if post.selftext.strip():
                    save_post(post, output_folder, filename_prefix="all")
                    post_count += 1

                    if post.author:
                        post_count += fetch_user_posts(post.author.name, post_date, output_folder, max_posts - post_count)

            break  # Exit the retry loop if successful
        except praw.exceptions.APIException as e:
            print(f"Rate limit exceeded: {e}. Retrying after a delay...")
            time.sleep(60)
            attempts += 1
        except Exception as e:
            print(f"Error fetching posts: {e}. Retrying...")
            time.sleep(30)
            attempts += 1

    print(f"\nTotal posts scraped: {post_count}")

# Main scraping logic
if __name__ == "__main__":
    for subreddit_name in subreddits:
        fetch_posts_from_subreddit(subreddit_name)

    fetch_posts_from_all()

    print("\nScraping complete! Text files saved in the respective folders.")

# NLTK setup
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Define input and output folders
folders = {
    "depression": {
        "input": "data/reddit_scraped_posts/depression",
        "output": "data/preprocessed_posts/depression"
    },
    "breastcancer": {
        "input": "data/reddit_scraped_posts/breastcancer",
        "output": "data/preprocessed_posts/breastcancer"
    },
    "standard": {
        "input": "data/reddit_scraped_posts/standard",
        "output": "data/preprocessed_posts/standard"
    }
}

# Ensure output folders exist
for category, paths in folders.items():
    os.makedirs(paths["output"], exist_ok=True)

def preprocess_text(text):
    """Preprocess text: tokenize, lowercase, remove stopwords, and apply stemming."""
    try:
        tokens = word_tokenize(text)
        tokens = [
            word.lower() for word in tokens 
            if word.isalnum() and word.lower() not in stop_words
        ]
        tokens = [stemmer.stem(word) for word in tokens]
        return " ".join(tokens)
    except Exception as e:
        print(f"Error preprocessing text: {e}")
        return ""

# Preprocess posts
for category, paths in folders.items():
    input_folder = paths["input"]
    output_folder = paths["output"]

    if not os.path.exists(input_folder):
        print(f"Input folder does not exist: {input_folder}")
        continue

    print(f"\nProcessing {category} posts...")

    for filename in os.listdir(input_folder):
        if filename.endswith(".txt"):
            input_filepath = os.path.join(input_folder, filename)
            output_filepath = os.path.join(output_folder, filename)

            try:
                with open(input_filepath, "r", encoding="utf-8") as infile:
                    text = infile.read()

                post_content = "\n".join(text.splitlines()[6:]).strip()

                if not post_content or '[removed]' in post_content.lower():
                    print(f"Skipping {filename} (empty or contains '[removed]')")
                    continue

                preprocessed_text = preprocess_text(post_content)

                with open(output_filepath, "w", encoding="utf-8") as outfile:
                    outfile.write(preprocessed_text)

                print(f"Processed {filename} into {output_folder}")
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

print("\nPreprocessing complete! Preprocessed files are saved in the respective output folders.")