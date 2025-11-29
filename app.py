from flask import Flask, request, jsonify
import os
import sys
import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException
import time
import base64
from datetime import datetime
import json
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import google.generativeai as genai

app = Flask(__name__)

# Force unbuffered output for logging
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
YOUR_EMAIL = "25f1002038@ds.study.iitm.ac.in"
YOUR_SECRET = "paper_tiger_2000"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL_NAME = os.environ.get("GEMINI_MODEL_NAME", "gemini-2.5-flash")


# ---------------------------------------------------------------------
# GEMINI SETUP
# ---------------------------------------------------------------------
def get_gemini_model():
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")
    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel(GEMINI_MODEL_NAME)


def call_gemini(prompt, timeout=120):
    """Call Gemini with timeout."""
    try:
        model = get_gemini_model()
        response = model.generate_content(
            prompt,
            request_options={"timeout": timeout}
        )
        return response.text
    except Exception as e:
        print(f"Gemini error: {e}", flush=True)
        return None


def extract_json(text):
    """Extract first JSON object from text."""
    if not text:
        return None
    
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    if start == -1:
        return None
    
    depth = 0
    in_string = False
    escape = False
    
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                json_str = text[start:i+1]
                try:
                    return json.loads(json_str)
                except:
                    return None
    return None


# ---------------------------------------------------------------------
# BROWSER
# ---------------------------------------------------------------------
def setup_browser():
    """Setup headless Chrome."""
    chrome_options = Options()
    chrome_options.binary_location = os.environ.get("CHROME_BIN", "/usr/bin/chromium")
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-gpu")
    
    service = Service(os.environ.get("CHROMEDRIVER_PATH", "/usr/bin/chromedriver"))
    return webdriver.Chrome(service=service, options=chrome_options)


def get_page_content(url):
    """Get page content with Selenium."""
    print(f"Fetching: {url}", flush=True)
    
    driver = None
    try:
        driver = setup_browser()
        driver.set_page_load_timeout(30)
        driver.get(url)
        
        # Wait for page to fully load
        time.sleep(5)
        
        # Try to get #result element first
        content = None
        try:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "result"))
            )
            element = driver.find_element(By.ID, "result")
            content = element.text
            print(f"Got content from #result", flush=True)
        except TimeoutException:
            print("#result not found, getting body", flush=True)
        
        # If #result failed or empty, try body
        if not content or len(content) < 50:
            body = driver.find_element(By.TAG_NAME, "body")
            content = body.text
            print(f"Got content from body", flush=True)
        
        html = driver.page_source
        print(f"Content length: {len(content)} chars", flush=True)
        
        return content, html
        
    except Exception as e:
        print(f"Selenium error: {e}, trying fallback", flush=True)
        
        # Fallback to requests
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            html = resp.text
            
            # Try to decode base64 content in atob()
            match = re.search(r"atob\([`'\"]([^`'\"]+)[`'\"]\)", html)
            if match:
                b64_str = match.group(1).replace("\n", "").replace(" ", "")
                try:
                    decoded = base64.b64decode(b64_str).decode("utf-8", errors="ignore")
                    print(f"Decoded base64: {len(decoded)} chars", flush=True)
                    return decoded, html
                except Exception as decode_err:
                    print(f"Base64 decode failed: {decode_err}", flush=True)
            
            # Extract text from HTML
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style tags
            for script in soup(["script", "style"]):
                script.decompose()
            
            text = soup.get_text()
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            text = '\n'.join(line for line in lines if line)
            
            print(f"Fallback extracted: {len(text)} chars", flush=True)
            return text, html
            
        except Exception as e2:
            print(f"Fallback also failed: {e2}", flush=True)
            return "", ""
            
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass


# ---------------------------------------------------------------------
# URL EXTRACTION
# ---------------------------------------------------------------------
def extract_urls_from_html(html, base_url):
    """Extract all potential data URLs from HTML."""
    urls = set()
    
    try:
        soup = BeautifulSoup(html, 'html.parser')
        
        # Find all links
        for tag in soup.find_all(['a', 'link', 'audio', 'video', 'source', 'img']):
            href = tag.get('href') or tag.get('src')
            if href:
                absolute_url = urljoin(base_url, href)
                urls.add(absolute_url)
        
        # Find relative paths mentioned in text (like "/demo-scrape-data")
        relative_pattern = r'(?:Scrape|Download|Get|Fetch)\s+([/\w\-\.?=@&]+)'
        for match in re.finditer(relative_pattern, html, re.IGNORECASE):
            path = match.group(1)
            if path.startswith('/') or path.startswith('http'):
                absolute_url = urljoin(base_url, path)
                urls.add(absolute_url)
    
    except Exception as e:
        print(f"URL extraction error: {e}", flush=True)
    
    return list(urls)


# ---------------------------------------------------------------------
# FILE OPERATIONS
# ---------------------------------------------------------------------
def download_file(url):
    """Download file and return content."""
    print(f"Downloading: {url}", flush=True)
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        
        # Return text if it's text-based
        if any(t in content_type.lower() for t in ["text", "json", "csv", "html"]):
            return resp.text, content_type, "text"
        else:
            return base64.b64encode(resp.content).decode(), content_type, "binary"
    except Exception as e:
        print(f"Download failed for {url}: {e}", flush=True)
        return None, None, None


def scrape_page(url):
    """Scrape webpage text."""
    print(f"Scraping: {url}", flush=True)
    try:
        driver = setup_browser()
        driver.get(url)
        time.sleep(2)
        text = driver.find_element(By.TAG_NAME, "body").text
        driver.quit()
        return text
    except Exception as e:
        print(f"Scraping failed: {e}", flush=True)
        try:
            resp = requests.get(url, timeout=30)
            soup = BeautifulSoup(resp.text, 'html.parser')
            return soup.get_text()
        except:
            return None


# ---------------------------------------------------------------------
# LLM PROCESSING
# ---------------------------------------------------------------------
def solve_quiz(quiz_content, quiz_url, html_content, extracted_urls):
    """Ask LLM to solve the quiz completely."""
    
    url_list = "\n".join([f"- {url}" for url in extracted_urls[:30]])
    
    prompt = f"""Analyze this quiz page and extract key information.

Quiz URL: {quiz_url}

Quiz Content:
{quiz_content}

URLs found in the page:
{url_list}

Tasks:
Read the question properly and try to understand what is being asked, read the question properly until you understand what is being asked
Dont assume, read the quiz content properly
1. Check if there's a direct answer shown in the content:
   - Look for: "answer": "X" or "answer: X" in the text
   - Example: If you see "answer": "anything you want", then answer IS "anything you want"
   - If found, extract it EXACTLY as written
   
2. Find data files or pages to access:
   - Look at the URLs list above
   - If quiz says "Scrape /some-path", find that URL in the list and add it to download_urls
   - If quiz mentions files like CSV, JSON, PDF, audio, images - find them in the URL list
   - Include ANY URL that needs to be accessed to solve the quiz
   
3. Determine if scraping is needed:
   - Set needs_scraping=true if the quiz says "scrape" 
   - Set needs_scraping=false for downloading files
   
4. Find submission URL:
   - Look for "POST to" or "submit to" in the content
   - Convert relative URLs to absolute (e.g., "/submit" becomes "[https://domain.com/submit](https://domain.com/submit)")

Return ONLY this JSON:
{{
  "download_urls": ["URLs that need to be accessed - from the list above"],
  "submit_url": "absolute URL for submission",
  "answer": "the direct answer if shown, otherwise null",
  "needs_scraping": true or false
}}

CRITICAL: If quiz says "Scrape /demo-scrape-data?email=...", you MUST find the matching URL in the list and add it to download_urls!
"""

    response = call_gemini(prompt, timeout=120)
    if response:
        return extract_json(response)
    return None


def process_file(file_url, content, content_type, file_type, quiz_context):
    """Process a downloaded file."""
    
    print(f"Processing {file_url} ({content_type}, {file_type})", flush=True)
    
    model = get_gemini_model()
    
    # Determine file type from URL and content type
    is_audio = "audio" in content_type or any(ext in file_url.lower() for ext in [".mp3", ".wav", ".ogg", ".opus", ".m4a", ".flac"])
    is_pdf = ".pdf" in file_url.lower() or "pdf" in content_type
    is_image = "image" in content_type or any(ext in file_url.lower() for ext in [".png", ".jpg", ".jpeg", ".gif", ".webp"])
    
    try:
        # Handle audio files with multimodal API
        if is_audio and file_type == "binary":
            raw = base64.b64decode(content)
            prompt = f"Transcribe this audio file exactly. Return only the transcription. Quiz context: {quiz_context[:500]}"
            response = model.generate_content(
                [prompt, {"mime_type": content_type, "data": raw}],
                request_options={"timeout": 120}
            )
            return response.text
        
        # Handle PDF files with multimodal API
        elif is_pdf and file_type == "binary":
            raw = base64.b64decode(content)
            prompt = f"Extract ALL text and data from this PDF. Preserve structure and numbers. Quiz context: {quiz_context[:500]}"
            response = model.generate_content(
                [prompt, {"mime_type": "application/pdf", "data": raw}],
                request_options={"timeout": 120}
            )
            return response.text
        
        # Handle images with multimodal API
        elif is_image and file_type == "binary":
            raw = base64.b64decode(content)
            prompt = f"Describe this image and extract any text, data, or numbers visible. Quiz context: {quiz_context[:500]}"
            response = model.generate_content(
                [prompt, {"mime_type": content_type, "data": raw}],
                request_options={"timeout": 120}
            )
            return response.text
        
        # Handle text-based files (CSV, JSON, TXT, HTML)
        elif file_type == "text":
            return content
        
        # Fallback for unrecognized formats
        else:
            print(f"Unrecognized file format, returning raw content", flush=True)
            if file_type == "text":
                return content
            else:
                return f"Binary file ({len(content)} bytes base64)"
            
    except Exception as e:
        print(f"File processing error: {e}", flush=True)
        import traceback
        traceback.print_exc()
    
    return None


def compute_final_answer(quiz_content, all_data):
    """Compute final answer from all collected data."""
    
    data_summary = "\n\n".join([f"Source: {url}\nData:\n{str(data)[:5000]}" for url, data in all_data.items() if data])
    
    prompt = f"""You are solving a data analysis quiz. Read the instructions extremely carefully and follow them exactly.

Quiz Instructions:
{quiz_content}

Available Data:
{data_summary}
Your task:
1. Parse the quiz instructions to understand EXACTLY what is being asked
2. Identify what operation needs to be performed (sum, count, filter, extract, transform, etc.)
3. Look for any conditions, filters, or thresholds mentioned in the instructions
4. Apply the correct operation to the data
5. Return the answer in the EXACT format requested (number, string, JSON object, list, etc.)

Important:
when given a csv cutoff value, it means that many values, eg if it is mentioned cutoff is 5, the calculation must be done with first 5 elementss
- Do NOT make assumptions about what operations to perform
- If instructions say "sum", calculate the sum
- If instructions say "count", count the items
- If instructions mention ANY filtering criteria (like a threshold, cutoff, condition), apply it BEFORE the operation
- If instructions say "where X > Y" or "above Z" or similar, filter accordingly
- If the data is CSV with one column of numbers, treat it as a list of numbers
- If the data is structured (JSON, CSV with headers), respect the structure
- Pay attention to data types - return numbers as numbers, strings as strings
- If transcribing audio, return the transcribed text
- If extracting from PDFs, return the extracted information

Return ONLY this JSON (no markdown, no explanation outside the JSON):
{{
  "answer": <your answer - number, string, object, or list>,
  "reasoning": "one sentence explaining what operation you performed"
}}
"""

    response = call_gemini(prompt, timeout=120)
    if response:
        result = extract_json(response)
        if result and "answer" in result:
            print(f"Reasoning: {result.get('reasoning', 'N/A')}", flush=True)
            return result["answer"]
    return None


def compute_correction(quiz_content, all_data, previous_answer, error_reason):
    """Ask LLM to correct the answer based on server feedback."""
    
    data_summary = "\n\n".join([f"Source: {url}\nData:\n{str(data)[:5000]}" for url, data in all_data.items() if data])
    
    prompt = f"""You previously submitted an answer that was incorrect.
    
Quiz Instructions:
{quiz_content}

Available Data:
{data_summary}

Previous Answer: {previous_answer}
Server Rejection Reason: {error_reason}

Task:
1. Analyze why the previous answer was wrong based on the server's reason.
2. Re-calculate the answer carefully.
3. Provide the corrected answer.

Return ONLY this JSON:
{{
  "answer": <new corrected answer>,
  "reasoning": "why the new answer is different"
}}
"""
    response = call_gemini(prompt, timeout=120)
    if response:
        result = extract_json(response)
        if result and "answer" in result:
            print(f"Correction Reasoning: {result.get('reasoning', 'N/A')}", flush=True)
            return result["answer"]
    return None


# ---------------------------------------------------------------------
# SUBMISSION
# ---------------------------------------------------------------------
def submit_answer(submit_url, email, secret, quiz_url, answer):
    """Submit answer to quiz endpoint."""
    payload = {
        "email": email,
        "secret": secret,
        "url": quiz_url,
        "answer": answer
    }
    print(f"Submitting to {submit_url}: answer={answer}", flush=True)
    try:
        resp = requests.post(submit_url, json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"Submission error: {e}", flush=True)
        # Check if we got a JSON response with details despite error code
        try:
            if hasattr(e, 'response') and e.response is not None:
                return e.response.json()
        except:
            pass
        return {"correct": False, "reason": str(e)}


# ---------------------------------------------------------------------
# MAIN SOLVER
# ---------------------------------------------------------------------
def solve_quiz_chain(initial_url, email, secret, max_iterations=10):
    """Solve chain of quizzes."""
    current_url = initial_url
    results = []
    start_time = time.time()
    
    for i in range(max_iterations):
        # Check 3 minute timeout
        elapsed = time.time() - start_time
        if elapsed > 170:  # 170 seconds = 2min 50sec, leave buffer
            print(f"Approaching 3 minute limit, stopping", flush=True)
            break
        
        print(f"\n{'='*60}", flush=True)
        print(f"Quiz {i+1}: {current_url}", flush=True)
        print(f"{'='*60}\n", flush=True)
        
        try:
            # Get quiz page
            quiz_content, html = get_page_content(current_url)
            if not quiz_content or len(quiz_content) < 20:
                print(f"Failed to get quiz content (got {len(quiz_content)} chars)", flush=True)
                results.append({"error": "No content", "url": current_url})
                break
            
            print(f"Content preview: {quiz_content[:300]}...", flush=True)
            
            # Extract all URLs from HTML
            extracted_urls = extract_urls_from_html(html, current_url)
            print(f"Extracted {len(extracted_urls)} URLs from page", flush=True)
            
            # Analyze quiz
            analysis = solve_quiz(quiz_content, current_url, html, extracted_urls)
            if not analysis:
                print("Failed to analyze quiz", flush=True)
                results.append({"error": "Analysis failed", "url": current_url})
                break
            
            print(f"Analysis: {json.dumps(analysis, indent=2)}", flush=True)
            
            # Collect data from external sources
            all_data = {}
            
            if analysis.get("download_urls"):
                for file_url in analysis["download_urls"]:
                    if analysis.get("needs_scraping"):
                        data = scrape_page(file_url)
                        if data:
                            all_data[file_url] = data
                    else:
                        content, ctype, ftype = download_file(file_url)
                        if content:
                            processed = process_file(file_url, content, ctype, ftype, quiz_content)
                            if processed:
                                all_data[file_url] = processed
            
            # Determine answer
            answer = None
            if analysis.get("answer") is not None:
                answer = analysis["answer"]
                print(f"Direct answer from quiz: {answer}", flush=True)
            elif all_data:
                answer = compute_final_answer(quiz_content, all_data)
                print(f"Computed answer from data: {answer}", flush=True)
            else:
                print("No answer could be determined", flush=True)
                results.append({"error": "No answer", "url": current_url})
                break
            
            if answer is None:
                print("Answer is None", flush=True)
                results.append({"error": "Answer is None", "url": current_url})
                break
            
            # Submit
            submit_url = analysis.get("submit_url")
            if not submit_url:
                # Fallback: try /submit on the same domain
                parsed = urlparse(current_url)
                submit_url = f"{parsed.scheme}://{parsed.netloc}/submit"
                print(f"No submit URL found, trying: {submit_url}", flush=True)
            
            # --- START RETRY LOGIC ---
            current_answer = answer
            retry_count = 0
            max_retries = 3
            result = {}
            
            while retry_count < max_retries:
                # Global timeout check
                if time.time() - start_time > 170:
                    break
                    
                result = submit_answer(submit_url, email, secret, current_url, current_answer)
                print(f"Result: {json.dumps(result, indent=2)}", flush=True)
                
                # If correct, or if we got a new URL to proceed to, stop retrying
                if result.get("correct") or (result.get("url") and result["url"] != current_url):
                    break
                
                # If wrong and no new URL, try to correct
                print(f"Answer wrong. Reason: {result.get('reason')}. Retrying...", flush=True)
                retry_count += 1
                
                new_answer = compute_correction(quiz_content, all_data, current_answer, result.get("reason"))
                if new_answer is not None and new_answer != current_answer:
                    current_answer = new_answer
                    print(f"Generated new corrected answer: {current_answer}", flush=True)
                else:
                    print("Could not generate a different answer.", flush=True)
                    break
            # --- END RETRY LOGIC ---
            
            quiz_status = "success" if result.get("correct") else "failure"
            print(f"QUIZ RESULT: {quiz_status.upper()} for {current_url}", flush=True)

            results.append({
                "quiz_number": i + 1,
                "url": current_url,
                "answer": current_answer,
                "correct": result.get("correct", False),
                "status": quiz_status,
                "reason": result.get("reason", "")
            })
            
            # Move to next quiz if provided
            if result.get("url"):
                current_url = result["url"]
                if result.get("delay"):
                    time.sleep(min(result["delay"], 5))
            else:
                print("No more quizzes", flush=True)
                break
                
        except Exception as e:
            print(f"Error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            results.append({"error": str(e), "url": current_url})
            break
    
    return results


# ---------------------------------------------------------------------
# FLASK ROUTES
# ---------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})


@app.route("/quiz", methods=["POST"])
def quiz_endpoint():
    print("\n" + "="*60, flush=True)
    print("New quiz request received", flush=True)
    print("="*60 + "\n", flush=True)
    
    try:
        data = request.get_json(force=True)
        if not isinstance(data, dict):
            return jsonify({"error": "Invalid JSON"}), 400
        
        email = data.get("email")
        secret = data.get("secret")
        url = data.get("url")
        
        if not all([email, secret, url]):
            return jsonify({"error": "Missing required fields"}), 400
        
        if secret != YOUR_SECRET or email != YOUR_EMAIL:
            return jsonify({"error": "Invalid credentials"}), 403
            
        # Run solver synchronously and return per-quiz results
        results = solve_quiz_chain(url, email, secret)

        return jsonify({
            "status": "completed",
            "timestamp": datetime.now().isoformat(),
            "results": results
        }), 200
        
    except Exception as e:
        print(f"Request error: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"Starting server on port {port}", flush=True)
    app.run(host="0.0.0.0", port=port)
