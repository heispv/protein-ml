import requests
from requests.adapters import HTTPAdapter, Retry
import re
from typing import Dict, Any, Iterator, Optional, Tuple
import logging

def create_session() -> requests.Session:
    retries = Retry(total=5, backoff_factor=0.25, status_forcelist=[500, 502, 503, 504])
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retries))
    logging.info("Created requests Session with retry configuration")
    return session

def get_next_link(headers: Dict[str, str]) -> Optional[str]:
    if "Link" in headers:
        re_next_link = re.compile(r'<(.+)>; rel="next"')
        match = re_next_link.match(headers["Link"])
        if match:
            return match.group(1)
    return None

def get_batch(batch_url: str, session: requests.Session) -> Iterator[Tuple[requests.Response, str]]:
    batch_count = 0
    while batch_url:
        batch_count += 1
        logging.info(f"Fetching batch {batch_count} from URL: {batch_url}")
        print(f"Fetching batch {batch_count}")
        response = session.get(batch_url)
        response.raise_for_status()
        total = response.headers["x-total-results"]
        yield response, total
        batch_url = get_next_link(response.headers)
    logging.info(f"Completed fetching {batch_count} batches")
    