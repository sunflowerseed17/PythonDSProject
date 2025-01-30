###############################################################################
#  IMPORTS
###############################################################################

import os
import re
import time
from datetime import datetime
import random
import praw
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords  # noqa: E402
from nltk.tokenize import word_tokenize  # noqa: E402
from nltk.stem import PorterStemmer  # noqa: E402

# Configure Reddit API credentials
# Removed the credentials for pushing to github so my reddit account does not get hacked :(
reddit = praw.Reddit(
    client_id="", 
    client_secret="",
    user_agent="",
    check_for_async=False
)

# Regex pattern to detect "I have been diagnosed with..."
search_pattern = re.compile(
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

    # Join back into a single string
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
        print(f"Error saving post {post.id}: {e}")

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
        print(f"Error saving preprocessed post {post.id}: {e}")

###############################################################################
# Fetch posts from the same user (within one month) if we haven't hit 1100
###############################################################################
def fetch_additional_posts_from_users(subreddit_name, user_set, already_fetched_count, max_posts):
    """
    If we haven't reached 1100 posts for a given subreddit (depression or breastcancer),
    look up the same authors we already grabbed and see if they have other posts
    in the same subreddit within the last month, that also match the pattern.

    Return the new total count of fetched posts for that subreddit.
    """

    raw_output_folder = f"data/reddit_scraped_posts/{subreddit_name}"
    preprocessed_output_folder = f"data/preprocessed_posts/{subreddit_name}"

    create_folder(raw_output_folder)
    create_folder(preprocessed_output_folder)

    # Calculate the timestamp for 1 month ago
    cutoff_1month_ago = time.time() - 30 * 24 * 60 * 60  # ~30 days in seconds

    count_fetched = already_fetched_count

    for username in user_set:
        if username is None or username.lower() == "deleted":
            continue
        if count_fetched >= max_posts:
            break

        try:
            # Access user by name
            redditor = reddit.redditor(username)
            for submission in redditor.submissions.new(limit=None):
                if count_fetched >= max_posts:
                    break

                # Stop if older than 1 month
                if submission.created_utc < cutoff_1month_ago:
                    # Because it's sorted by newest first, we can break on older posts
                    break

                # Check if it's in the same subreddit
                if str(submission.subreddit).lower() == subreddit_name.lower():
                    # Must have selftext, and match the pattern
                    if submission.selftext.strip() and re.search(search_pattern, submission.selftext):
                        save_post(submission, raw_output_folder, filename_prefix=subreddit_name)
                        save_preprocessed_post(submission, preprocessed_output_folder, filename_prefix=subreddit_name)
                        count_fetched += 1

                        # every 10 posts, do a 1-second pause:
                        if count_fetched % 10 == 0:
                            time.sleep(1)
                        else:
                            time.sleep(0.2)

        except Exception as e:
            # If the user is suspended, doesn't exist, etc., we get an error here
            print(f"Error fetching posts from user '{username}': {e}")
            continue

    return count_fetched


###############################################################################
# Fetch posts from a specific subreddit (depression or breastcancer)
###############################################################################
def fetch_posts_from_subreddit(subreddit_name, max_posts):
    """
    Fetch up to `max_posts` posts from the specified subreddit
    that match our search pattern, saving raw + preprocessed data.

    If we haven't reached `max_posts` after scanning the sub,
    attempt to fetch more from the same authors within the last month.

    Returns how many posts were finally saved for that subreddit.
    """
    print(f"\nFetching posts from r/{subreddit_name} with a max of {max_posts}...\n")

    # Define folders for raw and preprocessed output
    raw_output_folder = f"data/reddit_scraped_posts/{subreddit_name}"
    preprocessed_output_folder = f"data/preprocessed_posts/{subreddit_name}"

    create_folder(raw_output_folder)
    create_folder(preprocessed_output_folder)

    count_fetched = 0
    # We'll keep track of unique authors whose posts we found:
    authors_collected = set()

    try:
        subreddit = reddit.subreddit(subreddit_name)
        for post in subreddit.new(limit=None):
            if count_fetched >= max_posts:
                break

            if post.selftext.strip() and re.search(search_pattern, post.selftext):
                save_post(post, raw_output_folder, filename_prefix=subreddit_name)
                save_preprocessed_post(post, preprocessed_output_folder, filename_prefix=subreddit_name)

                count_fetched += 1
                if post.author is not None:
                    authors_collected.add(str(post.author))

                # Speed-up #1: shorter sleep each time
                if count_fetched % 10 == 0:
                    time.sleep(1)
                else:
                    time.sleep(0.2)

                # Print progress every 50 posts
                if count_fetched % 50 == 0:
                    print(f"  [r/{subreddit_name}] {count_fetched} posts saved so far...")

    except Exception as e:
        print(f"Error fetching posts from r/{subreddit_name}: {e}")

    print(f"Finished initial scraping of r/{subreddit_name}. Count so far: {count_fetched}")

    # If we haven't reached 1100, fetch from the same authors' 1-month posts
    if count_fetched < max_posts and authors_collected:
        print(f"Attempting to fetch additional posts from {len(authors_collected)} authors for r/{subreddit_name}...")
        new_count = fetch_additional_posts_from_users(
            subreddit_name=subreddit_name,
            user_set=authors_collected,
            already_fetched_count=count_fetched,
            max_posts=max_posts
        )
        count_fetched = new_count

    print(f"Final count for r/{subreddit_name}: {count_fetched}")
    return count_fetched


###############################################################################
# Fetch posts from random popular subreddits ("standard") 
###############################################################################
def fetch_posts_from_all(max_posts):
    """
    Fetch random posts from popular subreddits up to `max_posts` total,
    saving raw and preprocessed data. Return how many posts were saved.
    """
    print(f"\nFetching random posts (standard) with a max of {max_posts}...\n")

    # Define folders for raw and preprocessed output
    raw_output_folder = "data/reddit_scraped_posts/standard"
    preprocessed_output_folder = "data/preprocessed_posts/standard"
    create_folder(raw_output_folder)
    create_folder(preprocessed_output_folder)

    count_fetched = 0

    # Get a list of popular subreddits
    try:
        popular_subreddits = [sub.display_name for sub in reddit.subreddits.popular(limit=50)]
        random.shuffle(popular_subreddits)  # Shuffle them to get diversity
        print(f"Popular subreddits fetched (random sample): {popular_subreddits}")
    except Exception as e:
        print(f"Error fetching popular subreddits: {e}")
        return count_fetched

    for subreddit_name in popular_subreddits:
        if count_fetched >= max_posts:
            break

        try:
            subreddit = reddit.subreddit(subreddit_name)
            print(f"Fetching posts from r/{subreddit_name}...")

            # You could reduce the limit here too, e.g. limit=100, if speed is critical
            for post in subreddit.new(limit=50):
                if count_fetched >= max_posts:
                    break

                if post.selftext.strip():
                    save_post(post, raw_output_folder, filename_prefix="standard")
                    save_preprocessed_post(post, preprocessed_output_folder, filename_prefix="standard")

                    count_fetched += 1
                    # Speed-up #1: shorter sleep each time
                    if count_fetched % 10 == 0:
                        time.sleep(1)
                    else:
                        time.sleep(0.2)

                    # Print progress every 50 posts
                    if count_fetched % 50 == 0:
                        print(f"  [standard] {count_fetched} posts saved so far...")

        except Exception as e:
            print(f"Error fetching posts from r/{subreddit_name}: {e}")

    print(f"\nFinished scraping random subreddits (standard). Total fetched: {count_fetched}")
    return count_fetched

###############################################################################
# Utility: Print the total number of TXT files in each raw + preprocessed folder
###############################################################################
def print_file_counts_for_folder(folder_name):
    """
    Given a folder name (e.g. 'depression' or 'standard'), print
    how many .txt files are in the raw and preprocessed directories.
    """
    raw_path = f"data/reddit_scraped_posts/{folder_name}"
    pre_path = f"data/preprocessed_posts/{folder_name}"

    raw_count = 0
    pre_count = 0

    # Count .txt files in raw folder (if it exists)
    if os.path.exists(raw_path):
        raw_count = len([f for f in os.listdir(raw_path) if f.endswith('.txt')])

    # Count .txt files in preprocessed folder (if it exists)
    if os.path.exists(pre_path):
        pre_count = len([f for f in os.listdir(pre_path) if f.endswith('.txt')])

    print(f"\nFolder: {folder_name}")
    print(f" - Raw TXT files in '{raw_path}': {raw_count}")
    print(f" - Preprocessed TXT files in '{pre_path}': {pre_count}")

###############################################################################
# Main execution
###############################################################################
if __name__ == "__main__":
    max_posts_per_category = 1100

    # 1) Scrape r/depression (limit 1100, with possible 1-month user-based extension)
    depression_count = fetch_posts_from_subreddit("depression", max_posts_per_category)

    # 2) Scrape r/breastcancer (limit 1100, with possible 1-month user-based extension)
    breastcancer_count = fetch_posts_from_subreddit("breastcancer", max_posts_per_category)

    # 3) Scrape "standard" random popular subs (limit 1100; no user-based extension)
    standard_count = fetch_posts_from_all(max_posts_per_category)

    # Print overall summary
    print("\nSCRAPING COMPLETE!\n")

    # Print how many fetched in each category (based on internal counters)
    print("Fetched summary:")
    print(f" - depression: {depression_count}")
    print(f" - breastcancer: {breastcancer_count}")
    print(f" - standard: {standard_count}")

    # Now print how many files are actually in each folder
    print_file_counts_for_folder("depression")
    print_file_counts_for_folder("breastcancer")
    print_file_counts_for_folder("standard")

    print("\nAll done!")