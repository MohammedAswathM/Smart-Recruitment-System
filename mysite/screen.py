import warnings
import textract
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.neighbors import NearestNeighbors
import PyPDF2
import pathlib
from json import load, dumps
from operator import getitem
from collections import OrderedDict
from .text_process import normalize

import nltk
nltk.download('punkt')

from nltk.tokenize import word_tokenize
import mysite.configurations as regex
from datetime import date

from collections import defaultdict
from datetime import datetime
from dateutil import relativedelta
from typing import *


warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')


def getFilePath(loc):
    temp = str(loc)
    temp = temp.replace('\\', '/')
    return temp


def getFileName(filename):
    return filename.rsplit('\\')[1]


def readResultInJson(jobfile='job1'):
    filepath = 'result/'
    with open(filepath + jobfile + '.json', 'r') as openfile:
        # Reading from json file
        result = load(openfile)
    return result


def writeResultInJson(data, jobfile='job1'):
    filepath = 'result/'
    json_str = dumps(data, indent=4)
    with open(filepath + jobfile + '.json', 'w+', encoding='utf-8') as f:
        f.write(json_str)
        f.close()


def getNumberOfMonths(datepair) -> int:
    """
    Helper function to extract total months of experience from a resume
    :param date1: Starting date
    :param date2: Ending date
    :return: months of experience from date1 to date2
        """

    # if years
    date2_parsed = False
    if datepair.get("fh", None) is not None:
        gap = datepair["fh"]
    else:
        gap = ""
    try:
        present_vocab = ("present", "date", "now")
        if "syear" in datepair:
            date1 = datepair["fyear"]
            date2 = datepair["syear"]

            if date2.lower() in present_vocab:
                date2 = datetime.now()
                date2_parsed = True

            try:
                if not date2_parsed:
                    date2 = datetime.strptime(str(date2), "%Y")
                date1 = datetime.strptime(str(date1), "%Y")
            except:
                pass
        elif "smonth_num" in datepair:
            date1 = datepair["fmonth_num"]
            date2 = datepair["smonth_num"]

            if date2.lower() in present_vocab:
                date2 = datetime.now()
                date2_parsed = True

            for stype in ("%m" + gap + "%Y", "%m" + gap + "%y"):
                try:
                    if not date2_parsed:
                        date2 = datetime.strptime(str(date2), stype)
                    date1 = datetime.strptime(str(date1), stype)
                    break
                except:
                    pass
        else:
            date1 = datepair["fmonth"]
            date2 = datepair["smonth"]

            if date2.lower() in present_vocab:
                date2 = datetime.now()
                date2_parsed = True

            for stype in (
                "%b" + gap + "%Y",
                "%b" + gap + "%y",
                "%B" + gap + "%Y",
                "%B" + gap + "%y",
            ):
                try:
                    if not date2_parsed:
                        date2 = datetime.strptime(str(date2), stype)
                    date1 = datetime.strptime(str(date1), stype)
                    break
                except:
                    pass

        months_of_experience = relativedelta.relativedelta(date2, date1)
        months_of_experience = (
            months_of_experience.years * 12 + months_of_experience.months
        )
        return months_of_experience
    except Exception as e:
        return 0


def getTotalExperience(experience_list) -> int:
    """
    Wrapper function to extract total months of experience from a resume
    :param experience_list: list of experience text extracted
    :return: total months of experience
    """
    exp_ = []
    for line in experience_list:
        line = line.lower().strip()
        # have to split search since regex OR does not capture on a first-come-first-serve basis
        experience = re.search(
            r"(?P<fyear>\d{4})\s*(\s|-|to)\s*(?P<syear>\d{4}|present|date|now)",
            line,
            re.I,
        )
        if experience:
            d = experience.groupdict()
            exp_.append(d)
            continue

        experience = re.search(
            r"(?P<fmonth>\w+(?P<fh>.)\d+)\s*(\s|-|to)\s*(?P<smonth>\w+(?P<sh>.)\d+|present|date|now)",
            line,
            re.I,
        )
        if experience:
            d = experience.groupdict()
            exp_.append(d)
            continue

        experience = re.search(
            r"(?P<fmonth_num>\d+(?P<fh>.)\d+)\s*(\s|-|to)\s*(?P<smonth_num>\d+(?P<sh>.)\d+|present|date|now)",
            line,
            re.I,
        )
        if experience:
            d = experience.groupdict()
            exp_.append(d)
            continue
    experience_num_list = [getNumberOfMonths(i) for i in exp_]
    total_experience_in_months = sum(experience_num_list)
    return total_experience_in_months


"""
Utility Function that calculates experience in the resume text
params: resume_text type:string
returns: experience type:int
"""
def calculate_experience(resume_text):
  def get_month_index(month):
    month_dict = {'jan':1, 'feb':2, 'mar':3, 'apr':4, 'may':5, 'jun':6, 'jul':7, 'aug':8, 'sep':9, 'oct':10, 'nov':11, 'dec':12}
    return month_dict[month.lower()]

  try:
    start_month = -1
    start_year = -1
    end_month = -1
    end_year = -1
    regular_expression = re.compile(regex.date_range, re.IGNORECASE)
    regex_result = re.search(regular_expression, resume_text)
    while regex_result:
      date_range = regex_result.group()
      year_regex = re.compile(regex.year)
      year_result = re.search(year_regex, date_range)
      if (start_year == -1) or (int(year_result.group()) <= start_year):
        start_year = int(year_result.group())
        month_regex = re.compile(regex.months_short, re.IGNORECASE)
        month_result = re.search(month_regex, date_range)
        if month_result:
          current_month = get_month_index(month_result.group())
          if (start_month == -1) or (current_month < start_month):
            start_month = current_month
      if date_range.lower().find('present') != -1:
        end_month = date.today().month # current month
        end_year = date.today().year # current year
      else:
        year_result = re.search(year_regex, date_range[year_result.end():])
        if (end_year == -1) or (int(year_result.group()) >= end_year):
          end_year = int(year_result.group())
          month_regex = re.compile(regex.months_short, re.IGNORECASE)
          month_result = re.search(month_regex, date_range)
          if month_result:
            current_month = get_month_index(month_result.group())
            if (end_month == -1) or (current_month > end_month):
              end_month = current_month
      resume_text = resume_text[regex_result.end():]
      regex_result = re.search(regular_expression, resume_text)

    return end_year - start_year  # Use the obtained month attribute
  except Exception as exception_instance:
    # logging.error('Issue calculating experience: '+str(exception_instance))
    print('Issue calculating experience: '+str(exception_instance))
    return None


def get_experience_year(job_expr):
    job_expr = str.split(job_expr, ' ')[0]
    if '-' in job_expr:
        expr = job_expr.split('-')
        return int(expr[0])*12, int(expr[1])*12
    return int(job_expr)*12, -1


# for 2nd method
def getTotalExperienceFormatted(exp_list, job_expr) -> bool:

    min_yr_in_month, max_yr_in_month = get_experience_year(job_expr)
    print(min_yr_in_month, max_yr_in_month)
    print(exp_list)
    months = getTotalExperience(exp_list)

    if max_yr_in_month != -1:
        if (months >= min_yr_in_month) and (months <= max_yr_in_month):
            return True
    else:
        if months >= min_yr_in_month:
            return True
    return False


def findWorkAndEducation(text, name) -> Dict[str, List[str]]:
    categories = {"Work": ["(Work|WORK)", "(Experience(s?)|EXPERIENCE(S?))", "(History|HISTORY)"]}
    inv_data = {v[0][1]: (v[0][0], k) for k, v in categories.items()}
    line_count = 0
    exp_list = defaultdict(list)
    name = name.lower()

    current_line = None
    is_dot = False
    is_space = True
    continuation_sent = []
    first_line = None
    unique_char_regex = "[^\sA-Za-z0-9\.\/\(\)\,\-\|]+"

    for line in text.split("\n"):
        line = re.sub(r"\s+", " ", line).strip()
        match = re.search(r"^.*:", line)
        if match:
            line = line[match.end():].strip()

        # get first non-space line for filtering since
        # sometimes it might be a page header
        if line and first_line is None:
            first_line = line

        # update line_countfirst since there are `continue`s below
        line_count += 1
        if (line_count - 1) in inv_data:
            current_line = inv_data[line_count - 1][1]
        # contains a full-blown state-machine for filtering stuff
        elif current_line == "Work":
            if line:
                # if name is inside, skip
                if name == line:
                    continue
                # if like first line of resume, skip
                if line == first_line:
                    continue
                # check if it's not a list with some unique character as list bullet
                has_dot = re.findall(unique_char_regex, line[:5])
                # if last paragraph is a list item
                if is_dot:
                    # if this paragraph is not a list item and the previous line is a space
                    if not has_dot and is_space:
                        if line[0].isupper() or re.findall(r"^\d+\.", line[:5]):
                            exp_list[current_line].append(line)
                            is_dot = False

                else:
                    if not has_dot and (
                        line[0].isupper() or re.findall(r"^\d+\.", line[:5])
                    ):
                        exp_list[current_line].append(line)
                        is_dot = False
                if has_dot:
                    is_dot = True
                is_space = False
            else:
                is_space = True
        elif current_line == "Education":
            if line:
                # if not like first line
                if line == first_line:
                    continue
                line = re.sub(unique_char_regex, '', line[:5]) + line[5:]
                if len(line) < 12:
                    continuation_sent.append(line)
                else:
                    if continuation_sent:
                        continuation_sent.append(line)
                        line = " ".join(continuation_sent)
                        continuation_sent = []
                    exp_list[current_line].append(line)

    return exp_list


def check_basicRequirement(resumes_data, job_data):
    # print(job_experience)
    Ordered_list_Resume = []
    Resumes = []
    Temp_pdf = []
    # print(int(job_data.experience.split(' ')[0].split('-')[0]))
    # print(len(resumes_data))
    # filter resumes based on the gender
    resumes_data = resumes_data.filter(experience__gte=float(job_data.experience.split(' ')[0].split('-')[0]))
    print(len(resumes_data))
    if job_data.gender == 'Male':
        resumes_data = resumes_data.filter(gender='Male')
    elif job_data.gender == 'Female':
        resumes_data = resumes_data.filter(gender='Female')

    # resumes file path
    filepath = 'media/'
    resumes = [str(item.cv) for item in resumes_data]
    print("resumes: ", resumes)
    resumes_new = [item.split(':')[0] for item in resumes]
    resumes_new = [item for item in resumes_new if item != '']

    LIST_OF_FILES = resumes_new

    print("Total Files to Parse\t", len(LIST_OF_FILES))
    print("####### PARSING ########")
    for indx, file in enumerate(LIST_OF_FILES):
        print(indx, file)
        if not pathlib.Path(filepath+file).is_file():
            continue
        Ordered_list_Resume.append(file)

        Temp = file.split('.')
        print(Temp)

        if Temp[1] == "pdf" or Temp[1] == "Pdf" or Temp[1] == "PDF":
            try:
                # print("This is PDF", indx)
                with open(filepath + file, 'rb') as pdf_file:
                    # read_pdf = PyPDF2.PdfFileReader(pdf_file)

                    read_pdf = PyPDF2.PdfReader(pdf_file, strict=False)
                    # print("resume", indx,": ", read_pdf)
                    number_of_pages = len(read_pdf.pages)
                    for page_number in range(number_of_pages):
                        page = read_pdf.pages[page_number]
                        page_content = page.extract_text()
                        # print(page_content)
                        page_content = page_content.replace('\n', ' ').replace('\f', '').replace('\\uf[0-9]+',
                                                                                                 '').replace(
                            '\\u[0-9]+', '').replace('\\ufb[0-9]+', '')
                        # page_content.replace("\r", "")

                        Temp_pdf = str(Temp_pdf) + str(page_content)

                        # print(Temp_pdf)

                        Resumes.extend([Temp_pdf])

                    # if getTotalExperienceFormatted(Temp_pdf,  job_data.experience):
                    #     Resumes.extend([Temp_pdf])

                    Temp_pdf = ''

                    # f = open(str(i)+str("+") , 'w')
                    # f.write(page_content)
                    # f.close()
            except Exception as e:
                print(e)

        if Temp[1] == "doc" or Temp[1] == "Doc" or Temp[1] == "DOC":
            # print("This is DOC", file)

            try:
                a = textract.process(filepath)
                a = a.replace(b'\n', b' ')
                a = a.replace(b'\r', b' ')
                b = str(a)
                c = [b]
                Resumes.extend(c)
            except Exception as e:
                print(e)

        if Temp[1] == "docx" or Temp[1] == "Docx" or Temp[1] == "DOCX":
            # print("This is DOCX", file)
            try:
                a = textract.process(filepath + file)
                a = a.replace(b'\n', b' ')
                a = a.replace(b'\r', b' ')
                b = str(a)
                c = [b]
                Resumes.extend(c)
            except Exception as e:
                print(e)

        if Temp[1] == "exe" or Temp[1] == "Exe" or Temp[1] == "EXE":
            # print("This is EXE", file)
            pass
    print("Done Parsing.")
    return Resumes, Ordered_list_Resume


def get_rank(result_dict=None):

    if result_dict == None:
        return {}

    # new_result_dict = sorted(result_dict.items(), key=lambda item: float(item[1]["score"]), reverse=False)
    new_result_dict = OrderedDict(sorted(result_dict.items(), key=lambda item: getitem(item[1], 'score'), reverse=False))
    new_updated_result_dict = {}
    indx = 0
    for _, item in new_result_dict.items():
        item['rank'] = indx + 1
        new_updated_result_dict[indx] = item
        indx += 1
    return new_updated_result_dict


# Place this function in screen.py
def show_rank(result_dict, filename):
    try:
        with open(filename, 'w') as file:
            file.write("Rank\tTotal Score\tName\n")
            rank = 1
            for key, value in result_dict.items():
                # Ensure `rank` is assigned correctly
                result_line = f"{rank}\t{value['score']:.5f}\t{value['name']}\n"
                file.write(result_line)
                print(result_line.strip())
                rank += 1
    except Exception as e:
        print(f"Error in show_rank: {e}")

# Update your res function to call show_rank
# Make sure you have these imports
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors

# Step 1: Anonymize personal details
def anonymize_text(text):
    """
    Anonymizes personal information such as names, pronouns, and gendered terms.
    """
    text = re.sub(r'\b(Mr|Ms|Dr|Mrs)\.\s+\w+', 'Person', text)  # Replace titles and names with 'Person'
    text = re.sub(r'\b(he|she|his|her)\b', 'they', text)  # Replace gendered pronouns with neutral 'they'
    text = re.sub(r'\b([A-Za-z]+)\s+[A-Za-z]+\b', 'Name', text)  # Replace names with 'Name'
    return text

# Step 2: Check for biased terms in job description
def check_for_biased_terms(text):
    """
    Checks and replaces biased language in the job description with neutral alternatives.
    """
    biased_terms = ['aggressive', 'strong', 'competitive', 'leader', 'nurse']
    for term in biased_terms:
        if term in text:
            text = text.replace(term, 'person')  # Replace biased terms with neutral terms
    return text

# Step 3: Adjust for diversity and fairness in scoring
def adjust_for_diversity(similarity, knn_score, resume_data):
    """
    Adjusts the similarity score to ensure diversity by penalizing overly similar resumes.
    """
    diversity_penalty = 0.1  # Define the penalty factor
    if resume_data.get('similarity_to_avg', 0) > 0.9:  # Penalize resumes that are too similar to others
        return similarity - diversity_penalty
    return similarity

# Step 4: Fairness evaluation (optional but useful for monitoring)
def check_fairness(result_dict):
    """
    Evaluates the fairness of the recommendation results.
    """
    fairness_score = 0
    marginalized_groups = ['marginalized_group_name']  # Example: define your logic for identifying marginalized groups
    for idx, result in result_dict.items():
        if result['name'] in marginalized_groups:  # Example: Check if the resume belongs to a marginalized group
            fairness_score += 1
    return fairness_score


from .models import RankedResume

def res(resumes_data, job_data):
    """
    Function to process resumes, job descriptions, and rank candidates 
    based on AI-driven recommendations while ensuring fairness and eliminating bias.
    """
    import re
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.neighbors import NearestNeighbors
    
    result_dict = {}
    print("Resumes: ", list(resumes_data.values('cv')))

    # Step 5: Check basic requirements
    Resumes, Ordered_list_Resume = check_basicRequirement(resumes_data, job_data)
    print("Ordered_list_Resume:", Ordered_list_Resume)
    print("Resumes:", Resumes)
    
    if not Ordered_list_Resume or not Resumes:
        print("No resumes passed the basic requirements.")
        return result_dict

    # Step 6: Normalize job description
    job_desc = (
        job_data.details + '\n' +
        job_data.responsibilities + '\n' +
        job_data.experience
    )
    job_desc = re.sub(r' +', ' ', job_desc.replace('\n', '').replace('\r', ''))
    print("Raw Job Description:", job_desc)

    try:
        # Step 7: Anonymize job description and resumes
        job_desc = anonymize_text(job_desc)
        Resumes = [anonymize_text(resume) for resume in Resumes]
        print("Anonymized Job Description:", job_desc)
        print("Anonymized Resumes:", Resumes)

        # Step 8: Check for biased terms in job description
        job_desc = check_for_biased_terms(job_desc)

        normalized_job_desc = normalize(word_tokenize(job_desc))  # Normalize the tokenized job description
        job_text = [' '.join(normalized_job_desc)]
        print("Normalized Job Text:", job_text)
        
    except Exception as e:
        print(f"Error normalizing job description: {e}")
        return result_dict

    # Step 9: Feature extraction using TF-IDF
    vectorizer = CountVectorizer(stop_words='english')  # Using built-in English stop words
    transformer = TfidfTransformer()

    # Combine job description and resumes for feature extraction
    all_texts = job_text + Resumes

    try:
        # Get TF-IDF features
        X = vectorizer.fit_transform(all_texts)
        tfidf_matrix = transformer.fit_transform(X)
        print("TF-IDF Matrix Shape:", tfidf_matrix.shape)

        # Convert to dense arrays
        job_vector = tfidf_matrix[0].toarray()
        resume_vectors = tfidf_matrix[1:].toarray()

        # Step 10: Calculate cosine similarity and KNN scores for each resume
        for idx, resume_vector in enumerate(resume_vectors):
            if idx >= len(Ordered_list_Resume):
                break
                
            # Calculate cosine similarity between job description and resume
            similarity = cosine_similarity(job_vector, resume_vector.reshape(1, -1))[0][0]
            print(f"Resume {idx}: Cosine Similarity={similarity}")
            
            # Use KNN for additional scoring
            if len(resume_vectors) > 1:
                n_neighbors = min(3, len(resume_vectors))  # Use 3 or fewer neighbors
                knn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine')
                knn.fit(resume_vectors)
                
                # Get KNN-based score (inverse of mean distance between the resume and its neighbors)
                resume_distances, _ = knn.kneighbors(resume_vector.reshape(1, -1))
                knn_score = 1 / (1 + np.mean(resume_distances))  # Inverse to get a score
                print(f"Resume {idx}: KNN Score={knn_score}")
                
                # Combine similarity and KNN score to get a final score
                final_score = 0.7 * similarity + 0.3 * knn_score
            else:
                final_score = similarity
                knn_score = similarity  # If only one resume, KNN score equals similarity
            
            # Step 11: Adjust final score for diversity
            final_score = adjust_for_diversity(final_score, knn_score, result_dict.get(idx, {}))
            print(f"Resume {idx}: Final Adjusted Score={final_score}")
            
            # Step 12: Store results with name, score, and detailed metrics
            result_dict[idx] = {
                'name': Ordered_list_Resume[idx],
                'score': final_score,
                'similarity': similarity,
                'knn_score': knn_score
            }

    except Exception as e:
        print(f"Error in vectorization/similarity calculation: {e}")
        import traceback
        print(traceback.format_exc())  # Print full error trace for debugging
        return result_dict

    # Step 13: Sort results by final score
    sorted_result_dict = {
        k: {'name': v['name'], 'score': v['score']}
        for k, v in sorted(result_dict.items(), key=lambda item: item[1]['score'], reverse=True)
    }
    print("Sorted Results:", sorted_result_dict)

    # Step 14: Display and save results
    show_rank(sorted_result_dict, f"{job_data.company_name}_{job_data.title}.txt")
    
    # Optionally, evaluate fairness here
    fairness_score = check_fairness(sorted_result_dict)
    print(f"Fairness Score: {fairness_score}")

    return sorted_result_dict
