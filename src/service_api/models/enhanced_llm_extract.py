# utils/llm_extract.py

import json
import re
from llm.llm.gemini import RotateGemini




def _call_llm(messages, llm_client=None):
    """
    Unified call to LLM. If llm_client is provided, use it directly (callable).
    Otherwise create a RotateGemini fallback.
    """
    llm = llm_client if llm_client else RotateGemini(random_key=True, model_name="gemini-2.0-flash")

    try:
        resp = llm(messages)
        return resp
    except Exception as e:
        print(f"[LLM CALL ERROR] {e}")
        return None


def _clean_json_like(text: str) -> str:
    """Light clean: remove code fences, comments, smart quotes."""
    if not isinstance(text, str):
        text = str(text)
    s = text.strip()

    # Remove code fences
    s = re.sub(r"^```(?:json)?\s*", "", s)
    s = re.sub(r"\s*```$", "", s)

    # Remove JS-style comments
    s = re.sub(r"//.*?\n", "\n", s)
    s = re.sub(r"/\*[\s\S]*?\*/", "", s)

    # Smart quotes -> normal
    s = s.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

    # Decode common HTML entities
    s = s.replace("&nbsp;", " ").replace("&amp;", "&")

    return s


def _find_balanced_json_substrings(s: str):
    """Find substrings that look like {...} balanced JSON objects."""
    candidates = []
    stack = []
    start = None
    for i, ch in enumerate(s):
        if ch == "{":
            if not stack:
                start = i
            stack.append("{")
        elif ch == "}":
            if stack:
                stack.pop()
                if not stack and start is not None:
                    candidates.append(s[start:i + 1])
                    start = None
    return sorted(candidates, key=len, reverse=True)


def _try_fix_and_load(candidate: str):
    """Try to fix common JSON issues before json.loads."""
    work = candidate

    # Remove trailing commas
    work = re.sub(r",\s*([}\]])", r"\1", work)

    # Single-quoted keys → double quotes
    work = re.sub(r"(?P<prefix>[{\s,])'(?P<key>[^']+?)'\s*:", r'\g<prefix>"\g<key>":', work)

    # Single-quoted values → double quotes
    work = re.sub(r':\s*\'([^\']*?)\'', r': "\1"', work)

    try:
        return json.loads(work)
    except Exception:
        return None


def _parse_json_response(response):
    """
    Robust parser: try json.loads, then cleaning, then substring extraction.
    Always return dict (empty if fail).
    """
    if isinstance(response, dict):
        return {k: v.strip() if isinstance(v, str) else v for k, v in response.items()}

    if not isinstance(response, str):
        response = str(response)

    cleaned = _clean_json_like(response)

    # 1) Direct try
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return {k: v.strip() if isinstance(v, str) else v for k, v in data.items()}
    except Exception:
        pass

    # 2) Try balanced substrings
    for cand in _find_balanced_json_substrings(cleaned):
        if len(cand) < 20:
            continue
        parsed = _try_fix_and_load(cand)
        if parsed and isinstance(parsed, dict):
            return {k: v.strip() if isinstance(v, str) else v for k, v in parsed.items()}

    # 3) Aggressive fix on full text
    parsed = _try_fix_and_load(cleaned)
    if parsed and isinstance(parsed, dict):
        return {k: v.strip() if isinstance(v, str) else v for k, v in parsed.items()}

    print("Failed to parse JSON from response.")
    return {}




def trans(text, llm_client=None):
    """Translate content into English (plain text only)."""
    prompt = f"Translate this into plain English (no formatting, no explanation):\n\n{text}\n\nReturn only translated text."
    messages = [
        {"role": "system", "content": "You are a translation assistant. Output plain text only."},
        {"role": "user", "content": prompt},
    ]
    resp = _call_llm(messages, llm_client=llm_client)
    return resp.strip() if isinstance(resp, str) else str(resp) if resp else ""


def tach(text, llm_client=None):
    """Classify job text into exactly one predefined sector."""
    prompt = (
        f"Classify this text into exactly one sector from the list. "
        f"Return only the exact sector name (no quotes, no extra words):\n\n{text}\n\n"
        f"Predefined sectors:\n"
        f"Agriculture and Environment | Construction and Real Estate | Technology and IT | Manufacturing and Production | "
        f"Healthcare and Life Sciences | Education and Training | Finance and Insurance | Marketing and Advertising | "
        f"Retail, Sales, and Customer Service | Transportation and Logistics | Sports, Fitness, and Recreation | Media and Entertainment | "
        f"Hospitality and Tourism | Legal and Professional Services | Administrative | Nonprofit and Charitable Work | "
        f"Science and Research | Arts and Design | Human Resources (HR) | Others"
    )
    messages = [
        {"role": "system", "content": "Return only one sector name, exactly as in the list."},
        {"role": "user", "content": prompt},
    ]
    resp = _call_llm(messages, llm_client=llm_client)
    return resp.strip() if isinstance(resp, str) else str(resp).strip() if resp else "Others"


def get_job_details_from_gemini(html_content=None , infor_content=None, llm_client=None):
    llm = llm_client or RotateGemini(random_key=True, model_name="gemini-2.0-flash")
    response_cleaned = {}
    if infor_content:
        prompt = (
            f"You are given a block of unstructured text representing a job posting:\n\n{infor_content}\n\n"
            f"Your task is to extract structured information into JSON format. Do not invent fields that are not listed. "
            f"For each listed field, extract exact values from the content. If a value is missing or ambiguous, follow the rules below.\n\n"

            f"General Rules:\n"
            f"1. Extract only values that are clearly and explicitly present in the text. Do not guess.\n"
            f"2. Do not invent fields. Only extract the fields listed below.\n"
            f"3. Do not translate, paraphrase, interpret, or normalize values unless explicitly instructed.\n"
            f"4. If a value is missing, ambiguous, or cannot be extracted precisely, return 'Not available'.\n"
            f"5. All string values must preserve original phrasing, including symbols and accents.\n"
            f"6. Output must be **strictly valid JSON**: double quotes for all keys/values, no comments, no trailing commas.\n"
            f"7. Arrays must contain full objects or be omitted (do not include empty or null objects).\n\n"

            f"Inference Rules:\n"
            f"- Location (extract in vietnamese): If only a street name is given (e.g., 'Lê Thánh Tôn'), you may infer the most probable district and city in Vietnam. Mark such fields with \"Inferred\": true.\n"
            f"- Age: If only one value is given (e.g., 'trên 25 tuổi'), extract as Min_Age and leave Max_Age as 'Not available'.\n"
            f"- Salary: If a salary range is given (e.g., '15 - 30 triệu'), extract both min and max. If only one value appears (e.g., 'từ 15 triệu'), use it as Min_Salary and set Max_Salary to 'Not available'. Assume VND if currency is missing.\n"
            f"- Experience: Extract number of years as integer. If phrase is vague like 'Không yêu cầu', return 0.\n"
            f"- If field label is not exactly present (e.g., 'Ngành nghề', 'Cấp bậc'), you may infer the value from similar context and mark field 'Inferred': true.\n\n"
            f"- Age fields must either be integers or 'Not available'. Do not return phrases like '25+' or 'Dưới 35'.\n"

            f"Return the output in this exact JSON structure (no comments allowed):\n"
            f"{{\n"
            f"  \"Job_Type\": \"Extract from 'Loại hình làm việc' or 'Loại công việc'... Normalize to one of ['Full-time', 'Part-time', 'Remote', 'Freelance'], or 'Other'.\",\n"
            f"  \"Location_Detail\": [\n"
            f"    {{\n"
            f"      \"Detail_Address\": \"Extracted full address.\",\n"
            f"      \"Country\": \"Extracted country (e.g., 'Việt Nam').\",\n"
            f"      \"City\": \"Extracted city/province (e.g., 'Hồ Chí Minh', 'Tỉnh Thái Bình').\",\n"
            f"      \"District\": \"Extracted district-level location (e.g., 'Quận 1').\",\n"
            f"      \"Inferred\": true/false\n"
            f"    }}\n"
            f"    /* Repeat object above for each unique location */\n"
            f"  ],\n"
            f"  - Remove duplicate locations: If multiple entries have the same combination of 'Detail_Address', 'City', 'District', and 'Country', only include one unique object.\n"
            f"  - Do not include partially duplicated entries if a more complete version exists (e.g., skip an entry with only City if another entry already has both City and District).\n"
            f"  \"Job_Category\": \"Extracted from 'Ngành nghề'. Return original phrasing or 'Not available'.\",\n"
            f"  \"Contract_Type\": \"Extracted from 'Cấp bậc' or 'Vị trí'. Normalize to one of ['Intern', 'Staff', 'Supervisor', 'Consultant', 'Manager', 'Senior', 'Director', 'Executive', 'Freelance', 'Other'].\",\n"
            f"  \"Years_of_Experience\": \"Integer years extracted from 'Số năm kinh nghiệm'.\",\n"
            f"  \"Industry\": \"Extracted from 'Lĩnh vực'. Return exact text or 'Not available'.\",\n"
            f"  \"Min_Salary\": \"Minimum salary as integer and convert to VND (e.g., 15000000).\",\n"
            f"  \"Max_Salary\": \"Maximum salary as integer and convert to VND (e.g., 30000000).\",\n"
            f"  \"Sex_Requirement\": \"Extracted from 'Giới tính'. Return original value or 'Not available'.\",\n"
            f"  \"Min_Age\": \"Minimum age from 'Độ tuổi'.\",\n"
            f"  \"Max_Age\": \"Maximum age from 'Độ tuổi'.\"\n"
            f"}}"
        )

        message = [{'role': 'system','content': 'You must extract the content in JSON format.'},
        {'role': 'user','content': prompt}]
        response1 = llm(message)
        if isinstance(response1, dict):
            # If response is already a dict (e.g., when count_tokens=True), handle it directly
            return _parse_json_response(response1)
        elif not isinstance(response1, str):
            response1 = str(response1)
        response_cleaned = response1.strip().replace('```json', '').replace('```', '').strip()
    if html_content:
        prompt2 = (
            f"The following content contains a job posting:\n\n{html_content}\n\n"
            f"Your task is to extract structured information from this content and return it as a valid JSON object in english, following the exact field list and format described below.\n\n"

            f"Preprocessing:\n"
            f"- Remove '\\n' characters and redundant whitespace\n\n"

            f"Instructions:\n"
            f"- Only extract the specified fields listed below. Do not infer new fields or add extra data.\n"
            f"- For each field, return the value exactly as found in the content. If not found, return 'Not available'.\n"
            f"- For list fields, separate items with a semicolon ';'.\n"
            f"- Do not include explanations, markdown, or quotes around the JSON.\n\n"

            f"Field format:\n"
            f"{{\n"
            f"  \"Experience_Level\": \"One of: 'Entry-level', 'Mid-level', 'Senior-level', 'Manager', or 'Not available'\",\n"
            f"  \"Candidate_Experience_Requirements\": \"Extracted text related to required experience, if any, else 'Not available'.\",\n"
            f"  \"Candidate_degree_Requirements\": \"Extracted educational requirement from the content, else 'Not available'.\",\n"
            f"  \"Candidate_soft_skill_Requirements\": \"List of soft skills if explicitly mentioned, else 'Not available'.\",\n"
            f"  \"Candidate_technical_skill_Requirements\": \"List all mandatory technical skills stated. If not explicitly mentioned, infer basic role-related competencies (e.g., 'marketing knowledge (inferred)'). Do not include advanced technologies unless clearly stated.\",\n"
            f"  \"Benefits\": \"Extracted list of benefits provided in the post.\",\n"
            f"  \"Degree_Level\": \"One of: 'High School', 'Associate', 'Bachelor', 'Master', 'PhD', or 'Not available'\",\n"
            f"  \"Key_words\": \"Extract up to 15 relevant keywords from the job description, focused on job role relevance, every requirements, separated by semicolons.\"\n"
            f"}}\n\n"

            f"Return only the JSON object. Do not include explanations, commentary, or additional formatting."
        )

        message2 = [
            {
                'role': 'system',
                'content': 'You must extract the content in english and in JSON format.'
            },
            {
                'role': 'user',
                'content': prompt2
            }
        ]
        response2 = llm(message2)
        if isinstance(response2, dict):
            return _parse_json_response(response2)
        elif not isinstance(response2, str):
            response2 = str(response2)
        response_cleaned = response2.strip().replace('```json', '').replace('```', '').strip()   
        print(f"[LLM Response]: {response_cleaned}") 
    return _parse_json_response(response_cleaned)

def extract_job_sector(text: str) -> str:
    """
    Post-process or normalize the sector name returned by `tach`.
    Example: cleanup whitespace, standardize capitalization, etc.
    """
    if not text:
        return "Others"

    sector = text.strip()

    # Optional normalization rules
    mapping = {
        "IT": "Technology and IT",
        "Tech": "Technology and IT",
        "HR": "Human Resources (HR)",
    }
    return mapping.get(sector, sector)
