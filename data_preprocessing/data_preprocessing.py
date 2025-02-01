###############################################################################
#  IMPORTS
###############################################################################
import os
import re
import time
from datetime import datetime
import random
import logging
import praw
import nltk

# Download required NLTK resources
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

###############################################################################
# CONFIGURATION CONSTANTS
###############################################################################
# Reddit API credentials - replace with actual credentials, I don't want to share mine on the internet. :)
# If you don't want to create your own credentials, you can skip this and run with the scraped data
REDDIT_CLIENT_ID = ""
REDDIT_CLIENT_SECRET = ""
REDDIT_USER_AGENT = ""

# Directories for saving posts
RAW_POSTS_BASE_DIR = "data/reddit_scraped_posts"
PREPROCESSED_POSTS_BASE_DIR = "data/preprocessed_posts"

# Standard prefix for "standard" category posts
STANDARD_PREFIX = "standard"

# Maximum number of posts per category
MAX_POSTS_PER_CATEGORY = 1100

# Limits for popular subreddits and new posts fetched
POPULAR_SUBREDDITS_LIMIT = 50
NEW_POSTS_LIMIT = 50  # Limit for posts fetched from each subreddit in standard mode

# Time-related constants
ONE_MONTH_SECONDS = 30 * 24 * 60 * 60  # Approximate seconds in one month

# Sleep intervals and batch sizes for pacing API calls
POSTS_BATCH_SIZE = 10      # After every POSTS_BATCH_SIZE posts, use a longer sleep interval
SLEEP_SHORT = 0.2          # Short sleep interval (in seconds)
SLEEP_LONG = 1.0           # Long sleep interval (in seconds)
PROGRESS_INTERVAL = 50     # Log progress after every PROGRESS_INTERVAL posts

###############################################################################
# LOGGER CONFIGURATION
###############################################################################
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed output
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

###############################################################################
# Configure Reddit API credentials
###############################################################################
reddit = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT,
    check_for_async=False
)

# Regex pattern to detect diagnosis statements
SEARCH_PATTERN = re.compile(
    r"(i\s+(was|am|have been|got|recently got|just got|was just|had been|found out i\s+was|"
    r"was diagnosed as having|diagnosed as suffering from|got diagnosed as having|received a diagnosis of|"
    r"was told i\s+have|was informed i\s+have)\s+.*)",
    re.IGNORECASE
)

###############################################################################
# Preprocessing function
###############################################################################
def preprocess_text(text):
    """
    Convert to lowercase, remove URLs, remove Reddit usernames, tokenize,
    remove stopwords, and apply stemming.
    """
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove Reddit usernames (format: u/username)
    text = re.sub(r'u/\S+', '', text, flags=re.MULTILINE)

    # Tokenize
    tokens = word_tokenize(text)

    # Remove stopwords and keep alphabetic tokens only
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token.isalpha() and token not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    # Join tokens back into a single string
    preprocessed_text = ' '.join(tokens)
    return preprocessed_text

###############################################################################
# Folder creation helper
###############################################################################
def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

###############################################################################
# Save the raw scraped post
###############################################################################
def save_post(post, output_folder, filename_prefix):
    """
    Save the *raw* Reddit post to a file, using the category as part of the filename.
    """
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
    except Exception as e:
        logger.error("Error saving post %s: %s", post.id, e, exc_info=True)

###############################################################################
# Save the preprocessed post
###############################################################################
def save_preprocessed_post(post, output_folder, filename_prefix):
    """
    Save the *preprocessed* version of a Reddit post to a file.
    """
    preprocessed_text = preprocess_text(post.selftext)
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
            file.write(preprocessed_text)
    except Exception as e:
        logger.error("Error saving preprocessed post %s: %s", post.id, e, exc_info=True)

###############################################################################
# Fetch posts from the same user (within one month) if we haven't hit the limit
###############################################################################
def fetch_additional_posts_from_users(subreddit_name, user_set, already_fetched_count, max_posts):
    """
    Look up authors whose posts were initially fetched and see if they have additional
    posts in the same subreddit within the last month that match the search pattern.
    Return the updated count of fetched posts.
    """
    raw_output_folder = os.path.join(RAW_POSTS_BASE_DIR, subreddit_name)
    preprocessed_output_folder = os.path.join(PREPROCESSED_POSTS_BASE_DIR, subreddit_name)

    create_folder(raw_output_folder)
    create_folder(preprocessed_output_folder)

    count_fetched = already_fetched_count

    for username in user_set:
        if username is None or username.lower() == "deleted":
            continue
        if count_fetched >= max_posts:
            break

        try:
            redditor = reddit.redditor(username)
            for submission in redditor.submissions.new(limit=None):
                if count_fetched >= max_posts:
                    break

                # Break out if the post is older than one month
                if submission.created_utc < time.time() - ONE_MONTH_SECONDS:
                    break

                if str(submission.subreddit).lower() == subreddit_name.lower():
                    if submission.selftext.strip() and re.search(SEARCH_PATTERN, submission.selftext):
                        save_post(submission, raw_output_folder, filename_prefix=subreddit_name)
                        save_preprocessed_post(submission, preprocessed_output_folder, filename_prefix=subreddit_name)
                        count_fetched += 1

                        if count_fetched % POSTS_BATCH_SIZE == 0:
                            time.sleep(SLEEP_LONG)
                        else:
                            time.sleep(SLEEP_SHORT)
        except Exception as e:
            logger.error("Error fetching posts from user '%s': %s", username, e, exc_info=True)
            continue

    return count_fetched

###############################################################################
# Fetch posts from a specific subreddit (e.g., depression or breastcancer)
###############################################################################
def fetch_posts_from_subreddit(subreddit_name, max_posts):
    """
    Fetch up to `max_posts` posts from the specified subreddit that match the search pattern,
    saving both raw and preprocessed data. If the limit is not reached, attempt to fetch
    additional posts from the same authors within the last month.
    """
    logger.info("Fetching posts from r/%s with a max of %s...", subreddit_name, max_posts)

    raw_output_folder = os.path.join(RAW_POSTS_BASE_DIR, subreddit_name)
    preprocessed_output_folder = os.path.join(PREPROCESSED_POSTS_BASE_DIR, subreddit_name)

    create_folder(raw_output_folder)
    create_folder(preprocessed_output_folder)

    count_fetched = 0
    authors_collected = set()

    try:
        subreddit = reddit.subreddit(subreddit_name)
        for post in subreddit.new(limit=None):
            if count_fetched >= max_posts:
                break

            if post.selftext.strip() and re.search(SEARCH_PATTERN, post.selftext):
                save_post(post, raw_output_folder, filename_prefix=subreddit_name)
                save_preprocessed_post(post, preprocessed_output_folder, filename_prefix=subreddit_name)
                count_fetched += 1

                if post.author is not None:
                    authors_collected.add(str(post.author))

                if count_fetched % POSTS_BATCH_SIZE == 0:
                    time.sleep(SLEEP_LONG)
                else:
                    time.sleep(SLEEP_SHORT)

                if count_fetched % PROGRESS_INTERVAL == 0:
                    logger.info("  [r/%s] %s posts saved so far...", subreddit_name, count_fetched)

    except Exception as e:
        logger.error("Error fetching posts from r/%s: %s", subreddit_name, e, exc_info=True)

    logger.info("Finished initial scraping of r/%s. Count so far: %s", subreddit_name, count_fetched)

    if count_fetched < max_posts and authors_collected:
        logger.info("Attempting to fetch additional posts from %s authors for r/%s...", len(authors_collected), subreddit_name)
        count_fetched = fetch_additional_posts_from_users(
            subreddit_name=subreddit_name,
            user_set=authors_collected,
            already_fetched_count=count_fetched,
            max_posts=max_posts
        )

    logger.info("Final count for r/%s: %s", subreddit_name, count_fetched)
    return count_fetched

###############################################################################
# Fetch posts from random popular subreddits ("standard")
###############################################################################
def fetch_posts_from_all(max_posts):
    """
    Fetch random posts from popular subreddits up to `max_posts` total,
    saving raw and preprocessed data.
    """
    logger.info("Fetching random posts (standard) with a max of %s...", max_posts)

    raw_output_folder = os.path.join(RAW_POSTS_BASE_DIR, STANDARD_PREFIX)
    preprocessed_output_folder = os.path.join(PREPROCESSED_POSTS_BASE_DIR, STANDARD_PREFIX)
    create_folder(raw_output_folder)
    create_folder(preprocessed_output_folder)

    count_fetched = 0

    try:
        popular_subreddits = [sub.display_name for sub in reddit.subreddits.popular(limit=POPULAR_SUBREDDITS_LIMIT)]
        random.shuffle(popular_subreddits)
        logger.info("Popular subreddits fetched (random sample): %s", popular_subreddits)
    except Exception as e:
        logger.error("Error fetching popular subreddits: %s", e, exc_info=True)
        return count_fetched

    for subreddit_name in popular_subreddits:
        if count_fetched >= max_posts:
            break

        try:
            subreddit = reddit.subreddit(subreddit_name)
            logger.info("Fetching posts from r/%s...", subreddit_name)

            for post in subreddit.new(limit=NEW_POSTS_LIMIT):
                if count_fetched >= max_posts:
                    break

                if post.selftext.strip():
                    save_post(post, raw_output_folder, filename_prefix=STANDARD_PREFIX)
                    save_preprocessed_post(post, preprocessed_output_folder, filename_prefix=STANDARD_PREFIX)
                    count_fetched += 1

                    if count_fetched % POSTS_BATCH_SIZE == 0:
                        time.sleep(SLEEP_LONG)
                    else:
                        time.sleep(SLEEP_SHORT)

                    if count_fetched % PROGRESS_INTERVAL == 0:
                        logger.info("  [standard] %s posts saved so far...", count_fetched)

        except Exception as e:
            logger.error("Error fetching posts from r/%s: %s", subreddit_name, e, exc_info=True)

    logger.info("Finished scraping random subreddits (standard). Total fetched: %s", count_fetched)
    return count_fetched

###############################################################################
# Utility: Log the total number of TXT files in each raw and preprocessed folder
###############################################################################
def print_file_counts_for_folder(folder_name):
    """
    Given a folder name (e.g., 'depression' or 'standard'), log the number
    of .txt files in the raw and preprocessed directories.
    """
    raw_path = os.path.join(RAW_POSTS_BASE_DIR, folder_name)
    pre_path = os.path.join(PREPROCESSED_POSTS_BASE_DIR, folder_name)

    raw_count = len([f for f in os.listdir(raw_path) if f.endswith('.txt')]) if os.path.exists(raw_path) else 0
    pre_count = len([f for f in os.listdir(pre_path) if f.endswith('.txt')]) if os.path.exists(pre_path) else 0

    logger.info("Folder: %s", folder_name)
    logger.info(" - Raw TXT files in '%s': %s", raw_path, raw_count)
    logger.info(" - Preprocessed TXT files in '%s': %s", pre_path, pre_count)

###############################################################################
# Main execution
###############################################################################
if __name__ == "__main__":
    max_posts_per_category = MAX_POSTS_PER_CATEGORY
    # Scrape r/depression
    depression_count = fetch_posts_from_subreddit("depression", max_posts_per_category)
    # Scrape r/breastcancer
    breastcancer_count = fetch_posts_from_subreddit("breastcancer", max_posts_per_category)
    # Scrape "standard" random popular subreddits
    standard_count = fetch_posts_from_all(max_posts_per_category)

    logger.info("SCRAPING COMPLETE!")
    logger.info("Fetched summary:")
    logger.info(" - depression: %s", depression_count)
    logger.info(" - breastcancer: %s", breastcancer_count)
    logger.info(" - standard: %s", standard_count)
    logger.info("All done!")