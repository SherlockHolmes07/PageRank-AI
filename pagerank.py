import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    cont = dict()
    link_exists = len(corpus[page])

    if link_exists:
        #Choosing from all the pages of corpus
        for each_page in corpus:
            cont[each_page] = (1-damping_factor) / len(corpus)

         #Choosing from links on the page
        for link in corpus[page]:
            cont[link] += damping_factor / len(corpus[page])

    else:
        #If no links exists on the page
        for page in corpus:
            cont[page] = 1 / len(corpus)
    
    return cont


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """

    cont = dict()
    #initializing with Zero
    for key in corpus:
        cont[key] = 0

    #Picking up a page randomly and incrimenting it's rank
    page = random.choice(list(corpus.keys()))
    cont[page] += 1
    
    #Taking N samples
    for i in range(n):
        #Getting transition model for random page
        sample = transition_model(corpus,page,damping_factor)
        #Selecting a page from sample as per their probablity
        page = random.choices(list(sample.keys()), weights=sample.values(), k=1)[0]
        cont[page] += 1 #incrimenting its count

    #Getting the probablity by diving with n
    for key in cont:
        cont[key] = cont[key] / n

    return cont


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    ans = {}
    N = len(corpus)
    d = damping_factor
    #initializing with PR 1 / N
    for key in corpus:
        ans[key] = 1 / N

    while True:

        count = 0
        #For each key or Page in corpus
        for key in corpus:

            new = (1 - d) / N
            sum = 0

            #Traversing all the corpus and checking if the page has the link to the key and if it has then applying the formula
            for page in corpus:
                if key in corpus[page]:
                    sum += ans[page] / len(corpus[page])

                #As per question "A page that has no links at all should be interpreted as having one link for every page in the corpus"
                elif len(corpus[page]) == 0:
                    sum += ans[page] / len(corpus)

            sum = d * sum
            new += sum
            #Checking the terminating condition as per Que
            if abs(ans[key] - new) < 0.001:
                count += 1

            ans[key] = new 
        
        if count == N:
              return ans
  


if __name__ == "__main__":
    main()
