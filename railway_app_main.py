from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import time
import json
import base64
import requests
import pandas as pd
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from datetime import datetime
import sys
from urllib.parse import urlparse
import uuid
import io
import csv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# DataForSEO API credentials - use environment variables in production
API_LOGIN = os.environ.get('DATAFORSEO_LOGIN', 'admin@linkscience.ai')
API_PASSWORD = os.environ.get('DATAFORSEO_PASSWORD', '65573d10eab97090')
DATAFORSEO_BASE_URL = "https://api.dataforseo.com"

# Initialize the sentence transformer model
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Sentence transformer model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

# Store for batch jobs
batch_jobs = {}

def is_external_link(referring_url, target_url):
    """Check if referring_url is external link relative to target_url"""
    try:
        parsed_referring = urlparse(referring_url)
        parsed_target = urlparse(target_url)
        
        referring_domain = parsed_referring.netloc.lower().replace('www.', '')
        target_domain = parsed_target.netloc.lower().replace('www.', '')
        
        return referring_domain != target_domain
    except Exception as e:
        logger.error(f"Error checking if external link: {str(e)}")
        return True

class URLSimilarityCalculator:
    def __init__(self):
        self.content_cache = {}
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        }
        
    def fix_url(self, url):
        """Add scheme if missing and handle other URL issues"""
        if not url.startswith(('http://', 'https://')):
            return f'https://{url}'
        return url
            
    def get_page_content(self, url):
        """Fetch and extract main content from a URL with improved error handling"""
        if url in self.content_cache:
            return self.content_cache[url]
            
        try:
            fixed_url = self.fix_url(url)
            logger.info(f"Fetching content from: {fixed_url}")
            
            max_retries = 2
            retry_count = 0
            
            while retry_count <= max_retries:
                try:
                    response = requests.get(fixed_url, headers=self.headers, timeout=30, allow_redirects=True)
                    response.raise_for_status()
                    
                    content_type = response.headers.get('Content-Type', '').lower()
                    if not ('html' in content_type or 'text' in content_type):
                        logger.warning(f"URL content is not HTML (Content-Type: {content_type})")
                    
                    break
                    
                except requests.exceptions.Timeout:
                    retry_count += 1
                    if retry_count <= max_retries:
                        logger.info(f"Timeout occurred. Retrying ({retry_count}/{max_retries})...")
                        time.sleep(2)
                    else:
                        raise
                        
                except requests.exceptions.HTTPError as e:
                    if e.response.status_code in [403, 429]:
                        logger.warning(f"Access denied (status code: {e.response.status_code})")
                        self.headers['User-Agent'] = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
                        retry_count += 1
                        if retry_count <= max_retries:
                            logger.info(f"Retrying with different User-Agent ({retry_count}/{max_retries})...")
                            time.sleep(3)
                        else:
                            raise
                    else:
                        raise
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'iframe', 'noscript']):
                element.decompose()
            
            # Extract main content
            main_content = None
            for selector in ['main', 'article', 'div.content', 'div.main-content', 'div.article', '#content', '.post-content', '.entry-content']:
                elements = soup.select(selector)
                if elements:
                    main_content = elements[0]
                    break
            
            if main_content:
                paragraphs = main_content.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
                text = ' '.join([p.get_text(strip=True) for p in paragraphs])
            else:
                body = soup.body
                if body:
                    for unwanted in body.find_all(['nav', 'header', 'footer', 'aside', '.sidebar', '#sidebar']):
                        unwanted.decompose()
                    
                    paragraphs = body.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
                    text = ' '.join([p.get_text(strip=True) for p in paragraphs])
                else:
                    text = soup.get_text(separator=' ', strip=True)
            
            if len(text) < 200:
                logger.info(f"Extracted text is too short ({len(text)} chars), trying alternate method")
                text = ' '.join(soup.get_text(separator=' ', strip=True).split())
            
            if len(text) > 50000:
                text = text[:50000]
            
            if len(text) > 0:
                self.content_cache[url] = text
                logger.info(f"Successfully extracted {len(text)} characters from {fixed_url}")
                return text
            else:
                logger.error(f"Failed to extract text from {fixed_url}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching {url}: {str(e)}")
            return None

    def calculate_similarity(self, url1, url2):
        """Calculate similarity between two URLs"""
        if not model:
            logger.error("Model not loaded")
            return None
            
        try:
            content1 = self.get_page_content(url1)
            content2 = self.get_page_content(url2)
            
            if not content1 or not content2:
                return None
            
            embedding1 = model.encode([content1])[0]
            embedding2 = model.encode([content2])[0]
            
            similarity = cosine_similarity(
                embedding1.reshape(1, -1),
                embedding2.reshape(1, -1)
            )[0][0]
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error calculating similarity: {str(e)}")
            return None

class DataForSEOClient:
    def __init__(self, login, password):
        self.login = login
        self.password = password
        self.base_url = DATAFORSEO_BASE_URL
        self.auth_header = self._get_auth_header()
        
    def _get_auth_header(self):
        credentials = f"{self.login}:{self.password}"
        encoded_credentials = base64.b64encode(credentials.encode()).decode()
        return {"Authorization": f"Basic {encoded_credentials}"}
    
    def get_live_backlinks(self, target_domain, limit=100):
        """Get live backlinks for a target domain"""
        endpoint = f"{self.base_url}/v3/backlinks/backlinks/live"
        
        clean_domain = target_domain.replace("https://", "").replace("http://", "").rstrip("/")
        if clean_domain.startswith("www."):
            clean_domain = clean_domain[4:]
        
        post_data = [{
            "target": clean_domain,
            "limit": limit,
            "include_subdomains": True,
            "exclude_internal_backlinks": True,
            "backlinks_status_type": "live"
        }]
        
        logger.info(f"Fetching backlinks for {clean_domain}")
        
        try:
            response = requests.post(
                endpoint, 
                headers=self.auth_header,
                json=post_data
            )
            
            logger.info(f"DataForSEO API Response Status: {response.status_code}")
            
            if response.status_code != 200:
                logger.error(f"API Error: HTTP {response.status_code}")
                return []
            
            result = response.json()
            
            if result.get("status_code") != 20000:
                logger.error(f"API Error: {result.get('status_message')}")
                return []
            
            tasks = result.get("tasks", [])
            if not tasks:
                logger.info("No tasks found in API response")
                return []
            
            task_result = tasks[0].get("result", [])
            if not task_result:
                logger.info("No task results found in API response")
                return []
            
            items = task_result[0].get("items", [])
            if not items:
                logger.info("No backlink items found in API response")
                return []
            
            backlinks = []
            for item in items:
                referring_url = item.get("url_from")
                target_url = item.get("url_to", f"https://{clean_domain}")
                
                if is_external_link(referring_url, target_url):
                    backlink_data = {
                        "referring_url": referring_url,
                        "target_url": target_url,
                        "anchor_text": item.get("anchor", ""),
                        "external_domain": item.get("domain_from", ""),
                        "target_domain": clean_domain,
                        "first_seen": item.get("first_seen", ""),
                        "last_seen": item.get("last_seen", ""),
                        "page_authority": item.get("page_from_rank"),
                        "domain_authority": item.get("domain_from_rank")
                    }
                    backlinks.append(backlink_data)
                else:
                    logger.info(f"Skipping internal backlink: {referring_url} -> {target_url}")
            
            return backlinks
                
        except Exception as e:
            logger.error(f"Error fetching backlinks: {str(e)}")
            return []

# Initialize calculator and API client
calculator = URLSimilarityCalculator()
api_client = DataForSEOClient(API_LOGIN, API_PASSWORD)

@app.route('/', methods=['GET'])
def home():
    """Basic home page with API info"""
    return jsonify({
        'service': 'URL Similarity Calculator',
        'version': '1.0',
        'endpoints': {
            'single_similarity': '/calculate_similarity',
            'domain_backlinks': '/domain_backlinks',
            'batch_similarity': '/batch_similarity',
            'batch_status': '/batch_status/<job_id>',
            'batch_results': '/batch_results/<job_id>',
            'batch_csv': '/batch_csv/<job_id>'
        }
    })

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None
    })

@app.route('/calculate_similarity', methods=['POST'])
def calculate_similarity():
    """Calculate similarity between two URLs"""
    try:
        data = request.json
        external_url = data.get('external_url')
        target_url = data.get('target_url')
        
        if not external_url or not target_url:
            return jsonify({'error': 'Both external_url and target_url are required'}), 400
        
        logger.info(f"Calculating similarity: {external_url} vs {target_url}")
        
        similarity = calculator.calculate_similarity(external_url, target_url)
        
        if similarity is not None:
            return jsonify({
                'similarity': similarity,
                'external_url': external_url,
                'target_url': target_url,
                'status': 'success',
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'error': 'Could not calculate similarity',
                'external_url': external_url,
                'target_url': target_url,
                'status': 'error'
            }), 400
            
    except Exception as e:
        logger.error(f"Error in calculate_similarity: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/domain_backlinks', methods=['POST'])
def domain_backlinks():
    """Get backlinks for a domain and calculate similarity scores"""
    try:
        data = request.json
        target_domain = data.get('target_domain')
        limit = data.get('limit', 50)
        
        if not target_domain:
            return jsonify({'error': 'target_domain is required'}), 400
        
        logger.info(f"Fetching backlinks for domain: {target_domain}")
        
        # Get backlinks from DataForSEO
        backlinks = api_client.get_live_backlinks(target_domain, limit)
        
        if not backlinks:
            return jsonify({
                'target_domain': target_domain,
                'backlinks_found': 0,
                'results': [],
                'message': 'No backlinks found for this domain'
            })
        
        # Calculate similarity scores
        results = []
        for i, backlink in enumerate(backlinks):
            logger.info(f"Processing backlink {i+1}/{len(backlinks)}")
            
            similarity = calculator.calculate_similarity(
                backlink['referring_url'],
                backlink['target_url']
            )
            
            result = {
                'referring_url': backlink['referring_url'],
                'target_url': backlink['target_url'],
                'anchor_text': backlink['anchor_text'],
                'external_domain': backlink['external_domain'],
                'similarity': similarity,
                'status': 'success' if similarity is not None else 'error'
            }
            results.append(result)
            
            # Add small delay
            time.sleep(1)
        
        return jsonify({
            'target_domain': target_domain,
            'backlinks_found': len(backlinks),
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in domain_backlinks: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/batch_similarity', methods=['POST'])
def batch_similarity():
    """Start a batch similarity calculation job"""
    try:
        data = request.json
        url_pairs = data.get('url_pairs', [])
        
        if not url_pairs:
            return jsonify({'error': 'url_pairs is required and must be a non-empty list'}), 400
        
        for i, pair in enumerate(url_pairs):
            if not isinstance(pair, dict) or 'external_url' not in pair or 'target_url' not in pair:
                return jsonify({'error': f'Invalid format for URL pair at index {i}'}), 400
        
        job_id = str(uuid.uuid4())
        
        batch_jobs[job_id] = {
            'status': 'started',
            'total_pairs': len(url_pairs),
            'processed_pairs': 0,
            'results': [],
            'start_time': datetime.now().isoformat(),
            'end_time': None
        }
        
        logger.info(f"Started batch job {job_id} with {len(url_pairs)} URL pairs")
        
        def process_batch():
            try:
                batch_jobs[job_id]['status'] = 'processing'
                
                for i, pair in enumerate(url_pairs):
                    external_url = pair['external_url']
                    target_url = pair['target_url']
                    
                    logger.info(f"Job {job_id}: Processing pair {i+1}/{len(url_pairs)}")
                    
                    similarity = calculator.calculate_similarity(external_url, target_url)
                    
                    result = {
                        'external_url': external_url,
                        'target_url': target_url,
                        'similarity': similarity,
                        'status': 'success' if similarity is not None else 'error'
                    }
                    
                    batch_jobs[job_id]['results'].append(result)
                    batch_jobs[job_id]['processed_pairs'] = i + 1
                    
                    time.sleep(1)
                
                batch_jobs[job_id]['status'] = 'completed'
                batch_jobs[job_id]['end_time'] = datetime.now().isoformat()
                logger.info(f"Batch job {job_id} completed")
                
            except Exception as e:
                logger.error(f"Error in batch processing {job_id}: {str(e)}")
                batch_jobs[job_id]['status'] = 'error'
                batch_jobs[job_id]['error'] = str(e)
        
        import threading
        thread = threading.Thread(target=process_batch)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'job_id': job_id,
            'status': 'started',
            'total_pairs': len(url_pairs),
            'message': 'Batch processing started'
        })
        
    except Exception as e:
        logger.error(f"Error in batch_similarity: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/batch_status/<job_id>', methods=['GET'])
def batch_status(job_id):
    """Get status of a batch job"""
    if job_id not in batch_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = batch_jobs[job_id]
    
    progress_percentage = 0
    if job['total_pairs'] > 0:
        progress_percentage = (job['processed_pairs'] / job['total_pairs']) * 100
    
    return jsonify({
        'job_id': job_id,
        'status': job['status'],
        'total_pairs': job['total_pairs'],
        'processed_pairs': job['processed_pairs'],
        'progress_percentage': round(progress_percentage, 2),
        'start_time': job['start_time'],
        'end_time': job.get('end_time'),
        'error': job.get('error')
    })

@app.route('/batch_results/<job_id>', methods=['GET'])
def batch_results(job_id):
    """Get results of a batch job"""
    if job_id not in batch_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = batch_jobs[job_id]
    
    return jsonify({
        'job_id': job_id,
        'status': job['status'],
        'total_pairs': job['total_pairs'],
        'processed_pairs': job['processed_pairs'],
        'results': job['results'],
        'start_time': job['start_time'],
        'end_time': job.get('end_time')
    })

@app.route('/batch_csv/<job_id>', methods=['GET'])
def batch_csv(job_id):
    """Download batch results as CSV"""
    if job_id not in batch_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = batch_jobs[job_id]
    
    if job['status'] != 'completed':
        return jsonify({'error': 'Job not completed yet'}), 400
    
    output = io.StringIO()
    writer = csv.writer(output)
    
    writer.writerow(['External URL', 'Target URL', 'CS Score', 'Status'])
    
    for result in job['results']:
        writer.writerow([
            result['external_url'],
            result['target_url'],
            result['similarity'] if result['similarity'] is not None else 'N/A',
            result['status']
        ])
    
    output.seek(0)
    csv_data = output.getvalue()
    output.close()
    
    return send_file(
        io.BytesIO(csv_data.encode('utf-8')),
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'cosine_similarity_results_{job_id}.csv'
    )

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)