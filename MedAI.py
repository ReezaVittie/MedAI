"""
MedAI — Medical Assistant
Flask web application (converted from MedAI.html)

Dependencies:
    pip install flask requests

Run:
    python MedAI.py
Then open http://127.0.0.1:5000 in your browser.
"""

import hashlib
import html as html_mod
import json
import sys

# Raise recursion limit — gevent's SSL/socket monkey-patching and Python's re
# engine both create deep call stacks that exceed the default limit of 1000.
sys.setrecursionlimit(10000)
import os
import re
import time
import uuid
from datetime import datetime

import requests
from flask import Flask, Response, jsonify, render_template_string, request, stream_with_context, session, redirect, url_for

# ── Load .env file (keeps secrets out of source code) ────────────────────────
_env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

# ── Config ───────────────────────────────────────────────────────────────────
COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "")
if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY environment variable is not set. Add it in your Render dashboard.")
COHERE_MODEL   = "command-a-03-2025"
COHERE_URL     = "https://api.cohere.com/v2/chat"
# Use /tmp on Render (read-only filesystem) and local dir in dev
_DATA_DIR      = "/tmp" if os.environ.get("FLASK_ENV") == "production" else os.path.dirname(__file__)
DATA_FILE      = os.path.join(_DATA_DIR, "conversations.json")
USERS_FILE     = os.path.join(_DATA_DIR, "users.json")
MEDICAL_FILE   = os.path.join(_DATA_DIR, "medical_profiles.json")
EMERGENCY_FILE = os.path.join(_DATA_DIR, "emergency_services.json")

# ── Pre-compiled regex patterns (avoids recompiling on every request) ─────────
_RE_BOLD    = re.compile(r"\*\*(.+?)\*\*")
_RE_ITALIC  = re.compile(r"\*(.+?)\*")
_RE_CODE    = re.compile(r"`([^`]+)`")
_RE_H3      = re.compile(r"^### (.+)$", re.MULTILINE)
_RE_H2      = re.compile(r"^## (.+)$",  re.MULTILINE)
_RE_URGENT  = re.compile(r"\[URGENT:\s*(.+?)\]")
_RE_WARNING = re.compile(r"\[WARNING:\s*(.+?)\]")
_RE_BULLET  = re.compile(r"^[-•]\s+(.+)$", re.MULTILINE)
# _RE_UL_WRAP removed — (<li>[\s\S]*?</li>) caused RecursionError on long responses
_RE_NUMLIST = re.compile(r"^\d+\.\s+(.+)$", re.MULTILINE)
_RE_TAGS    = re.compile(r"<[^>]+>")

# ── Persistent HTTP session (reuses TCP connection to Cohere) ─────────────────
_http = requests.Session()
_http.headers.update({
    "Content-Type": "application/json",
    "Authorization": f"Bearer {COHERE_API_KEY}",
})

SYSTEM_PROMPT = """You are Remy, a specialized medical and safety assistant AI, not a general-purpose chatbot. Your primary role is to support user safety, situational awareness, and health-related decision-making without replacing medical professionals.

CORE PRINCIPLES:
- You are NOT a doctor and cannot diagnose conditions, prescribe medications, or guarantee outcomes
- Always include disclaimers like: "I'm not a doctor, but based on the information you've shared..."
- Prioritize user safety and encourage professional medical consultation for serious issues
- Be calm, supportive, reassuring, and transparent about your limitations

MEDICAL DATA HANDLING:
- Only access and use medical data with explicit user consent
- Never expose, log, or share medical data unnecessarily
- Use stored medical data only to improve safety recommendations, adjust guidance based on risk, provide contextual warnings, or assist in emergency situations
- When using medical data, briefly explain why: "Because you've indicated you have asthma..."

CONTEXT-AWARE INTELLIGENCE:
- Consider user medical profile, current situation/symptoms, location/time/environmental risks
- Apply clear logic: known condition + risky environment → warn user; known allergy + concerning symptoms → escalate guidance
- Cross-reference available context + medical data before generating guidance

EMERGENCY & ESCALATION:
- In high-risk situations: prioritize short, clear, calm instructions
- Encourage contacting emergency services or trusted contacts
- Avoid casual/humorous language in medical contexts
- If symptoms + conditions suggest immediate danger, escalate responsibly

PERSONALITY & TONE:
- Speak in a warm, gentle, friendly tone using "you" and "I" naturally
- Be comforting and reassuring while staying professional
- Show empathy first — acknowledge concerns before providing information
- Be transparent about limitations and focus on helping informed, safer decisions
- Use kind, patient-centered language: validate feelings, acknowledge worry, offer support

RESPONSE STRUCTURE:
1. Warm greeting and empathetic acknowledgment (1-2 sentences)
2. Clear explanation with disclaimer about not being a doctor
3. Practical next steps: home remedies, OTC options, lifestyle changes
4. When to see a doctor — be specific about red flags
5. Short reassuring closing with supportive language

FORMATTING RULES:
- Use **bold** for important terms or actions
- Use bullet lists for symptoms or steps
- Add warning blocks: [URGENT: text here] for emergencies, [WARNING: text here] for important cautions
- Never use jargon without explaining it
- Keep responses focused, warm, caring, and actionable

Always respect user autonomy and encourage professional care for anything serious, unclear, or persistent."""

# ── In-memory conversation cache ──────────────────────────────────────────────
# Loaded once from disk; all reads use the in-memory list directly.
# Writes update the list then flush to disk.

_convs: list = []
_loaded: bool = False


def _ensure_loaded() -> None:
    global _convs, _loaded
    if not _loaded:
        if os.path.exists(DATA_FILE):
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                _convs = json.load(f)
        _loaded = True


def _flush() -> None:
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(_convs, f, ensure_ascii=False, indent=2)

# ── User management ──────────────────────────────────────────────────────────
_users: dict = {}
_users_loaded: bool = False

def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def _ensure_users_loaded() -> None:
    global _users, _users_loaded
    if not _users_loaded:
        if os.path.exists(USERS_FILE):
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                _users = json.load(f)
        _users_loaded = True

def _flush_users() -> None:
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(_users, f, ensure_ascii=False, indent=2)

def register_user(email: str, password: str) -> tuple[bool, str]:
    _ensure_users_loaded()
    email = email.strip().lower()
    
    if not email or not password:
        return False, "Email and password are required"
    if "@" not in email:
        return False, "Invalid email address"
    if len(password) < 6:
        return False, "Password must be at least 6 characters"
    if email in _users:
        return False, "Email already registered"
    
    _users[email] = {
        "password": _hash_password(password),
        "created": int(time.time() * 1000)
    }
    _flush_users()
    return True, "Registration successful"

def login_user(email: str, password: str) -> tuple[bool, str]:
    _ensure_users_loaded()
    email = email.strip().lower()
    
    if email not in _users:
        return False, "Email not registered"
    if _users[email]["password"] != _hash_password(password):
        return False, "Invalid password"
    
    return True, email

# ── Emergency services management ─────────────────────────────────────────────
_emergency_services: dict = {}
_emergency_loaded: bool = False

def _ensure_emergency_loaded() -> None:
    """Load emergency services database"""
    global _emergency_services, _emergency_loaded
    if not _emergency_loaded:
        if os.path.exists(EMERGENCY_FILE):
            with open(EMERGENCY_FILE, "r", encoding="utf-8") as f:
                _emergency_services = json.load(f)
        else:
            # Initialize with default emergency services data
            # Keep default data in memory — do NOT write to disk on first load
            _emergency_services = {
                "South Africa": {
                    "Western Cape": {
                        "Cape Town": ["10177", "021 689 9531"],
                        "Stellenbosch": ["021 883 3191", "10177"],
                        "Paarl": ["021 863 1734", "10177"],
                        "George": ["044 802 6311", "10177"],
                        "Knysna": ["044 382 2311", "10177"]
                    },
                    "Gauteng": {
                        "Johannesburg": ["10177", "011 375 5911"],
                        "Pretoria": ["10177", "012 356 2128"],
                        "Sandton": ["011 463 7000", "10177"],
                        "Midrand": ["011 315 3000", "10177"],
                        "Centurion": ["012 663 1000", "10177"]
                    },
                    "KwaZulu-Natal": {
                        "Durban": ["10177", "031 361 5000"],
                        "Pietermaritzburg": ["033 845 7100", "10177"],
                        "Richards Bay": ["035 789 2111", "10177"],
                        "Newcastle": ["034 312 3444", "10177"]
                    },
                    "Eastern Cape": {
                        "Port Elizabeth": ["041 507 9111", "10177"],
                        "East London": ["043 702 2300", "10177"],
                        "Mthatha": ["047 532 4911", "10177"],
                        "Bhisho": ["040 609 5200", "10177"]
                    },
                    "Limpopo": {
                        "Polokwane": ["015 290 0360", "10177"],
                        "Thohoyandou": ["015 962 2294", "10177"],
                        "Tzaneen": ["015 307 2222", "10177"]
                    },
                    "Mpumalanga": {
                        "Nelspruit": ["013 755 2222", "10177"],
                        "Witbank": ["013 690 6222", "10177"],
                        "Middelburg": ["013 243 1222", "10177"]
                    },
                    "North West": {
                        "Rustenburg": ["014 590 1222", "10177"],
                        "Potchefstroom": ["018 294 1222", "10177"],
                        "Klerksdorp": ["018 462 1222", "10177"]
                    },
                    "Northern Cape": {
                        "Kimberley": ["053 839 1222", "10177"],
                        "Upington": ["054 332 1222", "10177"],
                        "Springbok": ["027 718 1222", "10177"]
                    },
                    "Free State": {
                        "Bloemfontein": ["051 430 1222", "10177"],
                        "Welkom": ["057 352 1222", "10177"],
                        "Bethlehem": ["058 303 1222", "10177"]
                    }
                },
                "United States": {
                    "California": {
                        "Los Angeles": ["911"],
                        "San Francisco": ["911"],
                        "San Diego": ["911"],
                        "Sacramento": ["911"]
                    },
                    "New York": {
                        "New York City": ["911"],
                        "Buffalo": ["911"],
                        "Albany": ["911"]
                    },
                    "Texas": {
                        "Houston": ["911"],
                        "Dallas": ["911"],
                        "Austin": ["911"],
                        "San Antonio": ["911"]
                    },
                    "Florida": {
                        "Miami": ["911"],
                        "Orlando": ["911"],
                        "Tampa": ["911"],
                        "Jacksonville": ["911"]
                    }
                },
                "United Kingdom": {
                    "England": {
                        "London": ["999", "112"],
                        "Manchester": ["999", "112"],
                        "Birmingham": ["999", "112"],
                        "Liverpool": ["999", "112"]
                    },
                    "Scotland": {
                        "Glasgow": ["999", "112"],
                        "Edinburgh": ["999", "112"],
                        "Aberdeen": ["999", "112"]
                    },
                    "Wales": {
                        "Cardiff": ["999", "112"],
                        "Swansea": ["999", "112"],
                        "Newport": ["999", "112"]
                    }
                },
                "Australia": {
                    "New South Wales": {
                        "Sydney": ["000"],
                        "Newcastle": ["000"],
                        "Wollongong": ["000"]
                    },
                    "Victoria": {
                        "Melbourne": ["000"],
                        "Geelong": ["000"],
                        "Ballarat": ["000"]
                    },
                    "Queensland": {
                        "Brisbane": ["000"],
                        "Gold Coast": ["000"],
                        "Cairns": ["000"]
                    }
                },
                "Canada": {
                    "Ontario": {
                        "Toronto": ["911"],
                        "Ottawa": ["911"],
                        "Hamilton": ["911"],
                        "London": ["911"]
                    },
                    "British Columbia": {
                        "Vancouver": ["911"],
                        "Victoria": ["911"],
                        "Surrey": ["911"]
                    },
                    "Alberta": {
                        "Calgary": ["911"],
                        "Edmonton": ["911"],
                        "Red Deer": ["911"]
                    }
                }
            }
        _emergency_loaded = True

def _flush_emergency() -> None:
    """Save emergency services to file"""
    with open(EMERGENCY_FILE, "w", encoding="utf-8") as f:
        json.dump(_emergency_services, f, ensure_ascii=False, indent=2)

def get_emergency_services(country: str = None, region: str = None, city: str = None) -> dict:
    """Get emergency services for specified location"""
    _ensure_emergency_loaded()
    
    if not country:
        return _emergency_services
    
    country_data = _emergency_services.get(country, {})
    if not region:
        return country_data
    
    region_data = country_data.get(region, {})
    if not city:
        return region_data
    
    return region_data.get(city, [])

def get_available_countries() -> list:
    """Get list of available countries"""
    _ensure_emergency_loaded()
    return list(_emergency_services.keys())

def get_available_regions(country: str) -> list:
    """Get list of available regions for a country"""
    _ensure_emergency_loaded()
    return list(_emergency_services.get(country, {}).keys())

def get_available_cities(country: str, region: str) -> list:
    """Get list of available cities for a country and region"""
    _ensure_emergency_loaded()
    return list(_emergency_services.get(country, {}).get(region, {}).keys())

# ── Flask app ─────────────────────────────────────────────────────────────────

def _ensure_medical_loaded() -> None:
    """Load medical profiles with validation and error recovery"""
    global _medical_profiles, _medical_loaded
    if not _medical_loaded:
        try:
            if os.path.exists(MEDICAL_FILE):
                with open(MEDICAL_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                # Validate data structure
                if isinstance(data, dict):
                    # Basic validation - ensure each profile has required structure
                    validated_data = {}
                    for email, profile in data.items():
                        if isinstance(profile, dict) and 'consent_given' in profile:
                            validated_data[email.lower()] = profile
                        else:
                            print(f"Skipping invalid medical profile for {email}")
                    
                    _medical_profiles = validated_data
                else:
                    print("Invalid medical profiles file structure, starting fresh")
                    _medical_profiles = {}
            else:
                _medical_profiles = {}
                
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading medical profiles: {e}")
            # Try to load from backup
            backup_file = MEDICAL_FILE + '.backup'
            if os.path.exists(backup_file):
                try:
                    with open(backup_file, "r", encoding="utf-8") as f:
                        _medical_profiles = json.load(f)
                    print("Loaded medical profiles from backup")
                except:
                    print("Backup also corrupted, starting fresh")
                    _medical_profiles = {}
            else:
                _medical_profiles = {}
        
        _medical_loaded = True

def _flush_medical() -> None:
    """Save medical profiles to file"""
    with open(MEDICAL_FILE, "w", encoding="utf-8") as f:
        json.dump(_medical_profiles, f, ensure_ascii=False, indent=2)

def get_medical_profile(email: str) -> dict:
    """Get user's medical profile with consent check"""
    _ensure_medical_loaded()
    email = email.strip().lower()
    return _medical_profiles.get(email, {})

def update_medical_profile(email: str, profile_data: dict) -> tuple[bool, str]:
    """Update user's medical profile with error handling"""
    try:
        _ensure_medical_loaded()
        email = email.strip().lower()
        
        if not email:
            return False, "Invalid email address"
        
        # Validate medical data structure
        valid_fields = {
            'allergies', 'chronic_conditions', 'medications', 'deficiencies', 
            'emergency_contacts', 'blood_type', 'insurance_info', 'consent_given'
        }
        
        # Only allow valid fields and require explicit consent
        if not profile_data.get('consent_given', False):
            return False, "Medical data consent is required"
        
        # Create backup of existing profile before updating
        existing_profile = _medical_profiles.get(email, {}).copy()
        
        filtered_data = {k: v for k, v in profile_data.items() if k in valid_fields}
        current_time = int(time.time() * 1000)
        filtered_data['updated'] = current_time
        filtered_data['last_modified'] = datetime.now().isoformat()
        
        # Set creation date if this is a new profile
        if not existing_profile:
            filtered_data['created'] = current_time
        
        _medical_profiles[email] = filtered_data
        
        # Attempt to save to file
        _flush_medical()
        
        # Verify the save was successful by re-reading
        _ensure_medical_loaded()
        if email not in _medical_profiles:
            # Restore backup if save failed
            if existing_profile:
                _medical_profiles[email] = existing_profile
                _flush_medical()
            return False, "Failed to save medical profile - data integrity check failed"
        
        return True, "Medical profile saved successfully"
        
    except Exception as e:
        # Attempt to restore backup on any error
        try:
            if existing_profile:
                _medical_profiles[email] = existing_profile
                _flush_medical()
        except:
            pass  # If backup restore fails, we can't do much
        return False, f"Error saving medical profile: {str(e)}"

def delete_medical_profile(email: str) -> tuple[bool, str]:
    """Delete user's medical profile with safeguards"""
    try:
        _ensure_medical_loaded()
        email = email.strip().lower()
        
        if not email:
            return False, "Invalid email address"
        
        if email not in _medical_profiles:
            return False, "Medical profile not found"
        
        # Create backup before deletion (in case of accidental deletion)
        deleted_profile = _medical_profiles[email].copy()
        deleted_profile['deleted_at'] = datetime.now().isoformat()
        deleted_profile['deleted_by'] = email
        
        # Remove from active profiles
        del _medical_profiles[email]
        
        # Save the changes
        _flush_medical()
        
        # Verify deletion was successful
        _ensure_medical_loaded()
        if email in _medical_profiles:
            return False, "Failed to delete medical profile - deletion verification failed"
        
        return True, "Medical profile deleted successfully"
        
    except Exception as e:
        return False, f"Error deleting medical profile: {str(e)}"

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-fallback-" + uuid.uuid4().hex)
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SECURE'] = os.environ.get("FLASK_ENV") == "production"

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>MedAI — Medical Assistant</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&family=DM+Serif+Display:ital@0;1&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet" />
<style>
/* ─── Reset ──────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
button { font-family: inherit; cursor: pointer; border: none; background: none; }
input, textarea { font-family: inherit; }

/* ─── Variables ──────────────────────────────────────── */
:root {
  --navy-950: #040c18;
  --navy-900: #071426;
  --navy-850: #0b1c34;
  --navy-800: #0f2340;
  --navy-700: #172e52;
  --navy-600: #1e3a66;
  --navy-400: #3d6494;
  --navy-300: #5a82b0;
  --navy-200: #8aaece;
  --navy-100: #c0d4e8;

  --accent:      #2f9eff;
  --accent-dim:  rgba(47,158,255,0.12);
  --accent-mid:  rgba(47,158,255,0.28);
  --accent-glow: #5ab4ff;

  --teal:     #1fd8c1;
  --teal-dim: rgba(31,216,193,0.1);

  --amber:     #f5a623;
  --amber-dim: rgba(245,166,35,0.1);

  --red:     #ef4444;
  --red-dim: rgba(239,68,68,0.1);

  --green:     #22c55e;
  --green-dim: rgba(34,197,94,0.08);

  --text-primary:   #deeaf6;
  --text-secondary: #6b90b4;
  --text-muted:     #344f67;

  --border-faint:  rgba(255,255,255,0.04);
  --border-subtle: rgba(255,255,255,0.07);
  --border-mid:    rgba(255,255,255,0.12);

  --font-sans:  'DM Sans', sans-serif;
  --font-serif: 'DM Serif Display', serif;
  --font-mono:  'JetBrains Mono', monospace;

  --sidebar-w: 285px;
  --topbar-h:  58px;
  --r:    12px;
  --r-sm:  8px;
  --r-lg: 16px;
}

/* ─── Base ───────────────────────────────────────────── */
html, body {
  height: 100%;
  overflow: hidden;
  background: var(--navy-950);
  color: var(--text-primary);
  font-family: var(--font-sans);
  font-size: 14px;
  line-height: 1.65;
  -webkit-font-smoothing: antialiased;
}

/* ─── App Shell ──────────────────────────────────────── */
.app { display: flex; height: 100vh; overflow: hidden; }

/* SIDEBAR */
.sidebar {
  width: var(--sidebar-w); min-width: var(--sidebar-w);
  height: 100%; background: var(--navy-900);
  border-right: 1px solid var(--border-faint);
  display: flex; flex-direction: column; overflow: hidden;
  transition: width .3s cubic-bezier(.4,0,.2,1), min-width .3s cubic-bezier(.4,0,.2,1);
  z-index: 50; position: relative;
}
.sidebar.collapsed { width: 0; min-width: 0; }

.sidebar-top { padding: 20px 16px 14px; border-bottom: 1px solid var(--border-faint); flex-shrink: 0; }
.brand { display: flex; align-items: center; gap: 11px; margin-bottom: 18px; text-decoration: none; }
.brand-icon {
  width: 38px; height: 38px; border-radius: var(--r-sm);
  background: linear-gradient(135deg, var(--accent), #1668b0);
  display: flex; align-items: center; justify-content: center;
  color: #fff; flex-shrink: 0; box-shadow: 0 0 22px rgba(47,158,255,.3);
}
.brand-name { font-family: var(--font-serif); font-size: 20px; font-weight: 400; color: var(--text-primary); line-height: 1; }
.brand-sub { font-size: 9.5px; font-weight: 500; color: var(--accent); letter-spacing: .1em; text-transform: uppercase; font-family: var(--font-mono); margin-top: 2px; }

.new-chat-btn {
  width: 100%; padding: 10px 14px; border-radius: var(--r-sm);
  border: 1px solid var(--accent-mid); background: var(--accent-dim);
  color: var(--accent-glow); font-size: 13px; font-weight: 600;
  display: flex; align-items: center; gap: 8px;
  transition: all .2s; margin-bottom: 12px; white-space: nowrap;
}
.new-chat-btn:hover { background: rgba(47,158,255,.2); border-color: rgba(47,158,255,.5); box-shadow: 0 0 18px rgba(47,158,255,.15); }

.search-wrap {
  display: flex; align-items: center; gap: 8px;
  background: var(--navy-850); border: 1px solid var(--border-faint);
  border-radius: var(--r-sm); padding: 8px 12px; color: var(--text-muted);
  transition: border-color .2s;
}
.search-wrap:focus-within { border-color: var(--border-subtle); }
.search-wrap input { background: none; border: none; outline: none; font-size: 12.5px; color: var(--text-primary); flex: 1; }
.search-wrap input::placeholder { color: var(--text-muted); }

.conv-section { flex: 1; overflow-y: auto; padding: 10px 10px 8px; }
.conv-section::-webkit-scrollbar { width: 3px; }
.conv-section::-webkit-scrollbar-thumb { background: var(--border-faint); border-radius: 2px; }

.section-lbl { font-size: 10px; font-weight: 600; letter-spacing: .09em; text-transform: uppercase; color: var(--text-muted); padding: 0 6px; margin-bottom: 7px; }
.conv-item {
  display: flex; align-items: center; gap: 9px;
  padding: 9px 10px; border-radius: var(--r-sm);
  cursor: pointer; transition: all .15s; margin-bottom: 2px;
  position: relative; border: 1px solid transparent;
}
.conv-item:hover { background: var(--navy-800); }
.conv-item.active { background: var(--accent-dim); border-color: var(--accent-mid); }
.conv-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--accent); flex-shrink: 0; opacity: .5; }
.conv-item.active .conv-dot { opacity: 1; }
.conv-body { flex: 1; min-width: 0; }
.conv-title { font-size: 12.5px; font-weight: 500; color: var(--text-primary); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; line-height: 1.3; margin-bottom: 2px; }
.conv-preview { font-size: 11px; color: var(--text-muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.conv-time { font-size: 9.5px; color: var(--text-muted); font-family: var(--font-mono); flex-shrink: 0; }
.conv-action { width: 22px; height: 22px; border-radius: 4px; color: var(--text-muted); display: none; align-items: center; justify-content: center; transition: all .15s; flex-shrink: 0; }
.conv-item:hover .conv-action { display: flex; }
.conv-action:hover { background: rgba(255,255,255,.08); color: var(--accent-glow); }
.conv-del { width: 22px; height: 22px; border-radius: 4px; color: var(--text-muted); display: none; align-items: center; justify-content: center; transition: all .15s; flex-shrink: 0; }
.conv-item:hover .conv-del { display: flex; }
.conv-del:hover { background: var(--red-dim); color: var(--red); }
.no-convs { text-align: center; padding: 28px 14px; color: var(--text-muted); font-size: 12px; line-height: 1.6; }

.sidebar-footer { padding: 12px 14px; border-top: 1px solid var(--border-faint); flex-shrink: 0; }
.disclaimer-pill { display: flex; align-items: center; gap: 6px; font-size: 10.5px; color: var(--amber); background: var(--amber-dim); border: 1px solid rgba(245,166,35,.18); border-radius: 20px; padding: 6px 12px; }

/* MAIN */
.main { flex: 1; display: flex; flex-direction: column; overflow: hidden; min-width: 0; }

.topbar {
  height: var(--topbar-h); background: rgba(7,20,38,.88);
  backdrop-filter: blur(12px); border-bottom: 1px solid var(--border-faint);
  display: flex; align-items: center; gap: 12px;
  padding: 0 18px; flex-shrink: 0; z-index: 10;
}
.toggle-btn { width: 34px; height: 34px; border-radius: var(--r-sm); border: 1px solid var(--border-faint); color: var(--text-secondary); display: flex; align-items: center; justify-content: center; transition: all .2s; flex-shrink: 0; }
.toggle-btn:hover { background: var(--navy-800); border-color: var(--border-subtle); color: var(--text-primary); }
.topbar-mid { display: flex; align-items: center; gap: 10px; flex: 1; min-width: 0; }
.topbar-title { font-size: 14px; font-weight: 600; color: var(--text-primary); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.online-badge { display: flex; align-items: center; gap: 5px; font-size: 10.5px; color: var(--green); font-family: var(--font-mono); background: var(--green-dim); border: 1px solid rgba(34,197,94,.18); border-radius: 20px; padding: 3px 10px; flex-shrink: 0; }
.online-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--green); animation: blink 2.5s ease-in-out infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:.35} }
.topbar-actions { display: flex; gap: 6px; }
.icon-btn { width: 34px; height: 34px; border-radius: var(--r-sm); border: 1px solid var(--border-faint); color: var(--text-secondary); display: flex; align-items: center; justify-content: center; transition: all .2s; }
.icon-btn:hover { background: var(--navy-800); color: var(--text-primary); border-color: var(--border-subtle); }

.chat-area { flex: 1; overflow-y: auto; scroll-behavior: smooth; }
.chat-area::-webkit-scrollbar { width: 4px; }
.chat-area::-webkit-scrollbar-thumb { background: var(--border-faint); border-radius: 2px; }
.msgs-wrap { max-width: 820px; margin: 0 auto; padding: 32px 24px 0; min-height: 100%; }

/* Welcome */
.welcome { display: flex; flex-direction: column; align-items: center; text-align: center; padding: 56px 0 36px; }
.intro-section { display: flex; flex-direction: column; align-items: center; text-align: center; margin-bottom: 20px; }
.orb-wrap { position: relative; width: 84px; height: 84px; display: flex; align-items: center; justify-content: center; margin-bottom: 28px; }
.orb-ring { position: absolute; inset: 0; border-radius: 50%; border: 1.5px solid var(--accent); opacity: 0; animation: ring 3.2s ease-out infinite; }
.orb-ring.d1 { animation-delay: 1.6s; }
@keyframes ring { 0%{transform:scale(.65);opacity:.7} 100%{transform:scale(1.7);opacity:0} }
.orb-core { width: 62px; height: 62px; border-radius: 50%; background: linear-gradient(140deg, var(--navy-700), var(--navy-850)); border: 1px solid var(--border-mid); display: flex; align-items: center; justify-content: center; color: var(--accent); position: relative; z-index: 1; box-shadow: 0 0 32px rgba(47,158,255,.22); }
.welcome-h { font-family: var(--font-serif); font-size: 32px; font-weight: 400; color: var(--text-primary); margin-bottom: 10px; line-height: 1.15; }
.welcome-intro { font-size: 14px; font-weight: 600; color: var(--accent); margin-bottom: 10px; letter-spacing: 0.02em; }
.welcome-p { font-size: 14px; color: var(--text-secondary); max-width: 460px; margin-bottom: 40px; line-height: 1.78; }
.starter-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 10px; width: 100%; max-width: 660px; }
.starter { padding: 14px 13px; border-radius: var(--r); border: 1px solid var(--border-faint); background: var(--navy-900); text-align: left; transition: all .2s; display: flex; align-items: flex-start; gap: 10px; }
.starter:hover { border-color: var(--accent-mid); background: var(--navy-850); transform: translateY(-2px); box-shadow: 0 8px 24px rgba(0,0,0,.35); }
.s-icon { font-size: 20px; flex-shrink: 0; line-height: 1; }
.s-lbl { font-size: 12.5px; font-weight: 500; color: var(--text-primary); line-height: 1.4; }
.s-lbl span { color: var(--text-secondary); font-weight: 400; }

/* Messages */
.msg-row { display: flex; gap: 12px; margin-bottom: 26px; animation: rise .3s ease; }
@keyframes rise { from{opacity:0;transform:translateY(9px)} to{opacity:1;transform:none} }
.msg-row.usr { flex-direction: row-reverse; }
.msg-av { width: 34px; height: 34px; border-radius: 10px; display: flex; align-items: center; justify-content: center; font-size: 13px; font-weight: 700; flex-shrink: 0; margin-top: 2px; }
.av-ai { background: linear-gradient(135deg, var(--accent), #1668b0); color: #fff; box-shadow: 0 0 14px rgba(47,158,255,.3); }
.av-user { background: linear-gradient(135deg, var(--teal), #0f9a88); color: #04131b; }
.msg-body { flex: 1; min-width: 0; max-width: calc(100% - 50px); }
.msg-row.usr .msg-body { display: flex; flex-direction: column; align-items: flex-end; }
.msg-head { display: flex; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 5px; }
.msg-row.usr .msg-head { flex-direction: row-reverse; }
.msg-who { font-size: 12px; font-weight: 600; letter-spacing: .04em; }
.copy-btn { width: 24px; height: 24px; border-radius: 4px; border: 1px solid var(--border-faint); background: rgba(255,255,255,.05); color: var(--text-muted); display: flex; align-items: center; justify-content: center; transition: all .2s; cursor: pointer; flex-shrink: 0; }
.copy-btn:hover { background: rgba(47,158,255,.12); color: var(--accent-glow); border-color: var(--accent-mid); }
.who-ai   { color: var(--accent-glow); }
.who-user { color: var(--teal); }
.msg-ts   { font-size: 10.5px; color: var(--text-muted); font-family: var(--font-mono); }
.bubble { padding: 14px 17px; border-radius: var(--r); font-size: 14px; line-height: 1.82; word-break: break-word; }
.b-ai { background: var(--navy-850); border: 1px solid var(--border-subtle); border-radius: 3px var(--r) var(--r) var(--r); color: var(--text-primary); }
.b-user { background: var(--accent-dim); border: 1px solid var(--accent-mid); border-radius: var(--r) 3px var(--r) var(--r); color: var(--text-primary); }
.bubble p { margin-bottom: 10px; }
.bubble p:last-child { margin-bottom: 0; }
.bubble strong { color: var(--accent-glow); font-weight: 600; }
.bubble em { color: var(--navy-100); font-style: italic; }
.bubble code { font-family: var(--font-mono); font-size: 12px; background: rgba(255,255,255,.07); padding: 2px 6px; border-radius: 4px; color: var(--teal); }
.bubble ul, .bubble ol { padding-left: 18px; margin: 8px 0; }
.bubble li { margin-bottom: 5px; }
.bubble h3 { font-size: 14px; font-weight: 600; color: var(--text-primary); margin: 14px 0 5px; border-bottom: 1px solid var(--border-faint); padding-bottom: 4px; }
.warn-block, .urgent-block { display: flex; gap: 8px; border-radius: var(--r-sm); padding: 10px 12px; margin-top: 12px; font-size: 12.5px; line-height: 1.55; }
.warn-block   { background: var(--amber-dim); border: 1px solid rgba(245,166,35,.22); color: var(--amber); }
.urgent-block { background: var(--red-dim);   border: 1px solid rgba(239,68,68,.22);  color: #f87171; }

/* Follow-up Questions */
.follow-up-questions { margin-top: 16px; padding-top: 16px; border-top: 1px solid var(--border-faint); }
.follow-up-title { font-size: 12px; font-weight: 600; color: var(--text-secondary); margin-bottom: 10px; text-transform: uppercase; letter-spacing: 0.5px; }
.follow-up-btn { display: block; width: 100%; padding: 10px 12px; margin-bottom: 6px; background: var(--navy-900); border: 1px solid var(--border-faint); border-radius: var(--r-sm); color: var(--text-primary); text-align: left; font-size: 13px; line-height: 1.4; cursor: pointer; transition: all .2s; }
.follow-up-btn:hover:not(:disabled) { background: var(--navy-800); border-color: var(--accent-mid); color: var(--accent-glow); }
.follow-up-btn:disabled { cursor: not-allowed; opacity: 0.6; }
.follow-up-btn:last-child { margin-bottom: 0; }

/* Symptom Map */
.symptom-card { margin: 0 auto 20px; padding: 20px; border: 1px solid var(--border-faint); border-radius: var(--r-lg); background: rgba(15,35,64,.88); max-width: 840px; }
.symptom-card h3 { margin-bottom: 10px; font-size: 15px; color: var(--text-primary); }
.symptom-card p { color: var(--text-secondary); font-size: 13px; margin-bottom: 16px; }
.symptom-map { display: grid; place-items: center; margin: 0 auto 12px; width: 100%; max-width: 300px; position: relative; }
.body-silhouette { width: 300px; height: 560px; background: linear-gradient(180deg, rgba(8,24,44,.98) 0%, rgba(16,38,72,.96) 50%, rgba(26,58,105,.92) 100%); border: 1px dashed rgba(255,255,255,.12); border-radius: 40px; position: relative; overflow: hidden; }
.body-silhouette::before { content: ''; position: absolute; top: 96px; left: 50%; transform: translateX(-50%); width: 2px; height: 344px; background: rgba(255,255,255,.08); }
.body-silhouette::after { content: ''; position: absolute; top: 276px; left: 16px; width: 268px; height: 1px; background: rgba(255,255,255,.08); opacity: .35; }
.body-part { position: absolute; border: 1px solid rgba(255,255,255,.18); background: rgba(255,255,255,.08); color: transparent; cursor: pointer; transition: all .2s; }
.body-part:hover, .body-part:focus { background: rgba(47,158,255,.28); border-color: rgba(47,158,255,.65); outline: none; color: var(--text-primary); }
.body-part::after { content: attr(data-region); position: absolute; left: 50%; top: 100%; transform: translate(-50%, 8px); white-space: nowrap; opacity: 0; pointer-events: none; font-size: 11px; color: var(--text-primary); background: rgba(7,20,38,.92); border: 1px solid rgba(255,255,255,.08); border-radius: 999px; padding: 4px 8px; transition: opacity .2s; }
.body-part:hover::after, .body-part:focus::after { opacity: 1; }
.body-part.active { background: rgba(47,158,255,.40); border-color: rgba(47,158,255,.75); }
.body-part.head { top: 18px; left: 50%; width: 60px; height: 62px; transform: translateX(-50%); border-radius: 50%; }
.body-part.left-eye { top: 28px; left: 135px; width: 8px; height: 8px; border-radius: 50%; }
.body-part.right-eye { top: 28px; left: 157px; width: 8px; height: 8px; border-radius: 50%; }
.body-part.nose { top: 38px; left: 145px; width: 10px; height: 10px; border-radius: 50%; }
.body-part.left-ear { top: 33px; left: 110px; width: 8px; height: 12px; border-radius: 4px; }
.body-part.right-ear { top: 33px; left: 182px; width: 8px; height: 12px; border-radius: 4px; }
.body-part.mouth { top: 58px; left: 144px; width: 12px; height: 6px; border-radius: 3px; }
.body-part.neck { top: 82px; left: 50%; width: 18px; height: 26px; transform: translateX(-50%); border-radius: 12px; }
.body-part.left-shoulder, .body-part.right-shoulder { top: 100px; width: 34px; height: 28px; border-radius: 20px 20px 14px 14px; }
.body-part.left-shoulder { left: 24px; }
.body-part.right-shoulder { right: 24px; }
.body-part.chest { top: 118px; left: 50%; width: 84px; height: 72px; transform: translateX(-50%); border-radius: 42px; }
.body-part.upper-back { top: 142px; left: 50%; width: 84px; height: 52px; transform: translateX(-50%); border-radius: 38px; }
.body-part.abdomen { top: 198px; left: 50%; width: 72px; height: 80px; transform: translateX(-50%); border-radius: 36px; }
.body-part.lower-back { top: 250px; left: 50%; width: 72px; height: 46px; transform: translateX(-50%); border-radius: 34px; }
.body-part.pelvis { top: 296px; left: 50%; width: 72px; height: 56px; transform: translateX(-50%); border-radius: 36px; }
.body-part.left-upper-arm, .body-part.right-upper-arm { top: 116px; width: 22px; height: 78px; border-radius: 18px; }
.body-part.left-upper-arm { left: 16px; }
.body-part.right-upper-arm { right: 16px; }
.body-part.left-forearm, .body-part.right-forearm { top: 194px; width: 18px; height: 72px; border-radius: 16px; }
.body-part.left-forearm { left: 14px; }
.body-part.right-forearm { right: 14px; }
.body-part.left-hand, .body-part.right-hand { top: 278px; width: 18px; height: 28px; border-radius: 14px; }
.body-part.left-hand { left: 14px; }
.body-part.right-hand { right: 14px; }
.body-part.left-thigh, .body-part.right-thigh { top: 302px; width: 28px; height: 78px; border-radius: 20px; }
.body-part.left-thigh { left: 72px; }
.body-part.right-thigh { right: 72px; }
.body-part.left-knee, .body-part.right-knee { top: 378px; width: 22px; height: 20px; border-radius: 14px; }
.body-part.left-knee { left: 74px; }
.body-part.right-knee { right: 74px; }
.body-part.left-calf, .body-part.right-calf { top: 400px; width: 20px; height: 72px; border-radius: 18px; }
.body-part.left-calf { left: 76px; }
.body-part.right-calf { right: 76px; }
.body-part.left-foot, .body-part.right-foot { top: 474px; width: 26px; height: 16px; border-radius: 14px; }
.body-part.left-foot { left: 70px; }
.body-part.right-foot { right: 70px; }
.region-question-panel { margin-top: 18px; padding: 16px 18px 18px; border: 1px solid var(--border-faint); border-radius: var(--r-sm); background: rgba(7,20,38,.9); }
.region-question-header { display: flex; flex-direction: column; gap: 6px; margin-bottom: 14px; }
.region-question-header span { font-size: 12px; color: var(--text-muted); }
.region-question-header strong { font-size: 15px; color: var(--text-primary); }
.question-row { display: grid; gap: 10px; margin-bottom: 12px; }
.question-label { font-size: 13px; color: var(--text-primary); }
.question-choice { display: flex; flex-wrap: wrap; gap: 8px; }
.question-choice button { padding: 9px 12px; border-radius: var(--r-sm); border: 1px solid var(--border-faint); background: var(--navy-900); color: var(--text-primary); font-size: 13px; cursor: pointer; transition: all .2s; }
.question-choice button.selected { background: rgba(47,158,255,.22); border-color: rgba(47,158,255,.55); color: var(--accent-glow); }
.question-choice button:hover { background: rgba(47,158,255,.14); border-color: rgba(47,158,255,.35); }
.question-input { width: 100%; min-height: 48px; resize: vertical; padding: 10px 12px; border-radius: var(--r-sm); border: 1px solid var(--border-faint); background: rgba(255,255,255,.03); color: var(--text-primary); outline: none; }
.question-actions { display: flex; justify-content: flex-end; gap: 10px; margin-top: 10px; }
.primary-btn, .secondary-btn { padding: 10px 14px; border-radius: var(--r-sm); font-size: 13px; font-weight: 600; border: 1px solid transparent; }
.primary-btn { background: var(--accent); color: #fff; }
.primary-btn:disabled { opacity: .5; cursor: not-allowed; background: var(--navy-800); }
.secondary-btn { background: rgba(255,255,255,.05); color: var(--text-primary); border-color: rgba(255,255,255,.08); }
.scroll-sidebar { position: fixed; right: 14px; top: 50%; transform: translateY(-50%); width: 18px; height: 52%; z-index: 20; }
.scroll-track { position: relative; width: 100%; height: 100%; background: rgba(255,255,255,.06); border-radius: 999px; border: 1px solid rgba(255,255,255,.08); }
.scroll-thumb { position: absolute; left: 2px; width: calc(100% - 4px); background: var(--accent); border-radius: 999px; cursor: pointer; transition: background .2s, transform .2s; }
.scroll-thumb:hover { background: rgba(47,158,255,.9); transform: scaleX(1.03); }
.scroll-thumb:focus { outline: 2px solid rgba(47,158,255,.6); outline-offset: 2px; }
@media (max-width: 900px) { .scroll-sidebar { right: 8px; } }
@media (max-width: 700px) { .symptom-card { margin-top: 18px; } .body-silhouette { width: 220px; height: 330px; } .body-part.head { width: 60px; height: 60px; } .body-part.chest { width: 86px; height: 74px; } .body-part.abdomen { width: 74px; height: 86px; } .scroll-sidebar { display: none; } }

/* Typing */
.typing-row { display: flex; gap: 12px; margin-bottom: 26px; align-items: center; }
.typing-bub { padding: 14px 18px; background: var(--navy-850); border: 1px solid var(--border-subtle); border-radius: 3px var(--r) var(--r) var(--r); display: flex; align-items: center; gap: 5px; }
.tdot { width: 7px; height: 7px; border-radius: 50%; background: var(--accent); display: inline-block; animation: tdot 1.4s ease-in-out infinite; }
.tdot:nth-child(2) { animation-delay:.2s; background: var(--accent-glow); }
.tdot:nth-child(3) { animation-delay:.4s; }
@keyframes tdot { 0%,100%{transform:scale(.7);opacity:.4} 50%{transform:scale(1);opacity:1} }

/* Input */
.input-zone { flex-shrink: 0; padding: 10px 24px 20px; position: relative; }
.input-zone::before { content:''; position:absolute; top:-52px; left:0; right:0; height:52px; background:linear-gradient(to top,var(--navy-950),transparent); pointer-events:none; }
.input-card { max-width: 820px; margin: 0 auto; background: var(--navy-850); border: 1px solid var(--border-mid); border-radius: var(--r-lg); padding: 14px 16px 10px; transition: border-color .2s, box-shadow .2s; }
.input-card:focus-within { border-color: rgba(47,158,255,.35); box-shadow: 0 0 0 3px rgba(47,158,255,.07); }
.input-inner { display: flex; align-items: flex-end; gap: 10px; }
.chat-ta { flex: 1; background: none; border: none; outline: none; color: var(--text-primary); font-size: 14px; line-height: 1.65; resize: none; min-height: 24px; max-height: 160px; overflow-y: auto; caret-color: var(--accent); }
.chat-ta::placeholder { color: var(--text-muted); }
.chat-ta::-webkit-scrollbar { width: 3px; }
.chat-ta::-webkit-scrollbar-thumb { background: var(--border-faint); }
.send-btn { width: 38px; height: 38px; border-radius: var(--r-sm); background: var(--accent); color: #fff; display: flex; align-items: center; justify-content: center; transition: all .2s; flex-shrink: 0; }
.send-btn:hover { background: var(--accent-glow); transform: scale(1.06); }
.send-btn:active { transform: scale(.94); }
.send-btn:disabled { opacity: .35; cursor: not-allowed; transform: none; }
.input-foot { display: flex; align-items: center; justify-content: space-between; margin-top: 8px; padding-top: 8px; border-top: 1px solid var(--border-faint); }
.input-hint { font-size: 11px; color: var(--text-muted); }
.input-hint kbd { background: var(--navy-800); border: 1px solid var(--border-subtle); border-radius: 3px; padding: 1px 5px; font-family: var(--font-mono); font-size: 10px; color: var(--text-secondary); }
.input-warn { font-size: 11px; color: var(--amber); opacity: .7; }

.overlay { display: none; position: fixed; inset: 0; background: rgba(0,0,0,.6); z-index: 40; }
@media(max-width:700px) {
  :root { --sidebar-w: 82vw; }
  .sidebar { position: fixed; top:0; left:0; height:100%; }
  .sidebar.collapsed { transform: translateX(-100%); width: var(--sidebar-w); min-width: var(--sidebar-w); }
  .overlay.show { display: block; }
  .starter-grid { grid-template-columns: 1fr 1fr; }
  .msgs-wrap { padding: 20px 14px 0; }
  .input-zone { padding: 10px 14px 18px; }
}
</style>
</head>
<body>
<div class="app" id="app">

  <!-- SIDEBAR -->
  <aside class="sidebar" id="sidebar">
    <div class="sidebar-top">
      <div class="brand">
        <div class="brand-icon">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
        </div>
        <div>
          <div class="brand-name">MedAI</div>
          <div class="brand-sub">Medical Assistant</div>
        </div>
      </div>
      <button class="new-chat-btn" onclick="createConversation()">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
        New Consultation
      </button>
      <button class="new-chat-btn" onclick="openMedicalProfile()" style="margin-top: 8px;">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="3"/><path d="M12 1v6m0 6v6m11-7h-6m-6 0H1"/></svg>
        Medical Profile
      </button>
      <button class="new-chat-btn emergency-btn" onclick="openEmergencyServices()" style="margin-top: 8px;">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/><circle cx="12" cy="12" r="3"/></svg>
        Emergency Services
      </button>
      <div class="search-wrap">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg>
        <input type="text" placeholder="Search consultations…" id="searchInput" oninput="filterConvs(this.value)" />
      </div>
    </div>
    <div class="conv-section">
      <div class="section-lbl">Recent Consultations</div>
      <div id="convList"></div>
    </div>
    <div class="sidebar-footer">
      <div class="disclaimer-pill">
        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
        Not a substitute for professional care
      </div>
    </div>
  </aside>

  <!-- MAIN -->
  <main class="main">
    <div class="topbar">
      <button class="toggle-btn" onclick="toggleSidebar()">
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><line x1="3" y1="6" x2="21" y2="6"/><line x1="3" y1="12" x2="21" y2="12"/><line x1="3" y1="18" x2="21" y2="18"/></svg>
      </button>
      <div class="topbar-mid">
        <span class="topbar-title" id="topTitle">New Consultation</span>
        <div class="online-badge"><span class="online-dot"></span>AI Online</div>
      </div>
      <div class="topbar-actions">
        <button class="icon-btn" title="Clear chat" onclick="clearChat()">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6M14 11v6M9 6V4h6v2"/></svg>
        </button>
        <button class="icon-btn" title="Sign out" onclick="window.location.href='/signout'">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4M16 17l5-5-5-5M21 12H9"/></svg>
        </button>
      </div>
    </div>

    <div class="chat-area" id="chatArea">
      <div class="msgs-wrap" id="msgsWrap">
        <div class="intro-section">
          <div class="orb-wrap">
            <div class="orb-ring"></div><div class="orb-ring d1"></div>
            <div class="orb-core">
              <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
            </div>
          </div>
          <p class="welcome-intro">Hi, I’m Remy.</p>
          <h2 class="welcome-h">How can I help you today?</h2>
          <p class="welcome-p">Describe your symptoms, ask about a condition, or get clear guidance on medications and treatments. I'll respond like a knowledgeable doctor would.</p>
        </div>
        <div class="symptom-card" aria-label="Symptom map control">
          <h3>Visual Symptom Map</h3>
          <p>Click a body region to answer a few focused follow-up questions. After you answer, Remy will summarize your responses and send them into the chat as structured context.</p>
          <div class="symptom-map" role="group" aria-label="Interactive symptom map">
            <div class="body-silhouette" id="bodySilhouette">
              <button type="button" class="body-part head" data-region="Head" aria-label="Head" onclick="selectRegion('Head')"></button>
              <button type="button" class="body-part left-eye" data-region="Left eye" aria-label="Left eye" onclick="selectRegion('Left eye')"></button>
              <button type="button" class="body-part right-eye" data-region="Right eye" aria-label="Right eye" onclick="selectRegion('Right eye')"></button>
              <button type="button" class="body-part nose" data-region="Nose" aria-label="Nose" onclick="selectRegion('Nose')"></button>
              <button type="button" class="body-part left-ear" data-region="Left ear" aria-label="Left ear" onclick="selectRegion('Left ear')"></button>
              <button type="button" class="body-part right-ear" data-region="Right ear" aria-label="Right ear" onclick="selectRegion('Right ear')"></button>
              <button type="button" class="body-part mouth" data-region="Mouth" aria-label="Mouth" onclick="selectRegion('Mouth')"></button>
              <button type="button" class="body-part neck" data-region="Neck" aria-label="Neck" onclick="selectRegion('Neck')"></button>
              <button type="button" class="body-part left-shoulder" data-region="Left shoulder" aria-label="Left shoulder" onclick="selectRegion('Left shoulder')"></button>
              <button type="button" class="body-part right-shoulder" data-region="Right shoulder" aria-label="Right shoulder" onclick="selectRegion('Right shoulder')"></button>
              <button type="button" class="body-part chest" data-region="Chest" aria-label="Chest" onclick="selectRegion('Chest')"></button>
              <button type="button" class="body-part upper-back" data-region="Upper back" aria-label="Upper back" onclick="selectRegion('Upper back')"></button>
              <button type="button" class="body-part abdomen" data-region="Abdomen" aria-label="Abdomen" onclick="selectRegion('Abdomen')"></button>
              <button type="button" class="body-part lower-back" data-region="Lower back" aria-label="Lower back" onclick="selectRegion('Lower back')"></button>
              <button type="button" class="body-part pelvis" data-region="Pelvis" aria-label="Pelvis" onclick="selectRegion('Pelvis')"></button>
              <button type="button" class="body-part left-upper-arm" data-region="Left upper arm" aria-label="Left upper arm" onclick="selectRegion('Left upper arm')"></button>
              <button type="button" class="body-part right-upper-arm" data-region="Right upper arm" aria-label="Right upper arm" onclick="selectRegion('Right upper arm')"></button>
              <button type="button" class="body-part left-forearm" data-region="Left forearm" aria-label="Left forearm" onclick="selectRegion('Left forearm')"></button>
              <button type="button" class="body-part right-forearm" data-region="Right forearm" aria-label="Right forearm" onclick="selectRegion('Right forearm')"></button>
              <button type="button" class="body-part left-hand" data-region="Left hand" aria-label="Left hand" onclick="selectRegion('Left hand')"></button>
              <button type="button" class="body-part right-hand" data-region="Right hand" aria-label="Right hand" onclick="selectRegion('Right hand')"></button>
              <button type="button" class="body-part left-thigh" data-region="Left thigh" aria-label="Left thigh" onclick="selectRegion('Left thigh')"></button>
              <button type="button" class="body-part right-thigh" data-region="Right thigh" aria-label="Right thigh" onclick="selectRegion('Right thigh')"></button>
              <button type="button" class="body-part left-knee" data-region="Left knee" aria-label="Left knee" onclick="selectRegion('Left knee')"></button>
              <button type="button" class="body-part right-knee" data-region="Right knee" aria-label="Right knee" onclick="selectRegion('Right knee')"></button>
              <button type="button" class="body-part left-calf" data-region="Left calf" aria-label="Left calf" onclick="selectRegion('Left calf')"></button>
              <button type="button" class="body-part right-calf" data-region="Right calf" aria-label="Right calf" onclick="selectRegion('Right calf')"></button>
              <button type="button" class="body-part left-foot" data-region="Left foot" aria-label="Left foot" onclick="selectRegion('Left foot')"></button>
              <button type="button" class="body-part right-foot" data-region="Right foot" aria-label="Right foot" onclick="selectRegion('Right foot')"></button>
            </div>
          </div>
          <div class="region-question-panel" id="regionQuestionPanel" hidden>
            <div class="region-question-header">
              <span>Selected body region</span>
              <strong id="selectedRegionName">Chest</strong>
            </div>
            <div id="regionQuestions"></div>
            <div class="question-actions">
              <button type="button" class="secondary-btn" onclick="cancelRegionSelection()">Cancel</button>
              <button type="button" class="primary-btn" id="submitRegionBtn" onclick="submitRegionQuestions()" disabled>Send symptom details</button>
            </div>
          </div>
        </div>
        <div class="welcome" id="welcome">
          <div class="starter-grid">
            <button class="starter" onclick="useStarter('I\'ve had a persistent headache for 3 days — it\'s dull but constant. What could be causing it and what should I do?')">
              <span class="s-icon">🧠</span>
              <span class="s-lbl">Persistent headache<br><span>3 days, dull &amp; constant</span></span>
            </button>
            <button class="starter" onclick="useStarter('I have a sore throat, mild fever (38°C) and fatigue since yesterday. Could this be strep throat or just a viral cold?')">
              <span class="s-icon">🤒</span>
              <span class="s-lbl">Sore throat &amp; fever<br><span>Strep or cold?</span></span>
            </button>
            <button class="starter" onclick="useStarter('What are the early warning signs of type 2 diabetes I should watch out for?')">
              <span class="s-icon">🩸</span>
              <span class="s-lbl">Diabetes warning signs<br><span>Early symptoms</span></span>
            </button>
            <button class="starter" onclick="useStarter('I feel chest tightness and shortness of breath when climbing stairs. Should I be concerned about my heart?')">
              <span class="s-icon">❤️</span>
              <span class="s-lbl">Chest tightness<br><span>When active</span></span>
            </button>
            <button class="starter" onclick="useStarter('What is the difference between ibuprofen and paracetamol (acetaminophen), and which should I take for pain and fever?')">
              <span class="s-icon">💊</span>
              <span class="s-lbl">Ibuprofen vs Paracetamol<br><span>Which to take?</span></span>
            </button>
            <button class="starter" onclick="useStarter('I\'ve been feeling persistently anxious and having trouble sleeping for the past few weeks. What are my options?')">
              <span class="s-icon">🧘</span>
              <span class="s-lbl">Anxiety &amp; poor sleep<br><span>Weeks-long struggle</span></span>
            </button>
          </div>
        </div>
        <div id="msgList"></div>
        <div class="typing-row" id="typingRow" style="display:none">
          <div class="msg-av av-ai">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>
          </div>
          <div class="typing-bub">
            <span class="tdot"></span><span class="tdot"></span><span class="tdot"></span>
          </div>
        </div>
        <div id="anchor" style="height:24px"></div>
      </div>
    </div>

    <div class="input-zone">
      <div class="input-card">
        <div class="input-inner">
          <textarea id="chatInput" class="chat-ta" placeholder="Describe your symptoms or ask a medical question…" rows="1" onkeydown="handleKey(event)" oninput="grow(this)"></textarea>
          <button class="send-btn" id="sendBtn" onclick="sendMessage()">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>
          </button>
        </div>
        <div class="input-foot">
          <span class="input-hint">Press <kbd>Enter</kbd> to send · <kbd>Shift+Enter</kbd> for new line</span>
          <span class="input-warn">⚠️ Always see a doctor for serious symptoms</span>
        </div>
      </div>
    </div>
  </main>
  <div class="scroll-sidebar" role="navigation" aria-label="Page scroll controls">
    <div class="scroll-track" id="scrollTrack" tabindex="0" aria-orientation="vertical" role="slider" aria-label="Scroll chat" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0">
      <div class="scroll-thumb" id="scrollThumb" tabindex="0" role="button" aria-label="Drag to scroll chat"></div>
    </div>
  </div>
</div>

<div class="overlay" id="overlay" onclick="toggleSidebar()"></div>

<script>
/* ── State ── */
let conversations = [];
let activeId = null;
let busy = false;
let sidebarOpen = true;

const SYMPTOM_REGIONS = {
  "Head": [
    { text: "Do you feel pain, pressure, or sensitivity?", type: "choice", options: ["Pain", "Pressure", "Sensitivity", "None"] },
    { text: "Is the pain steady, throbbing, or sharp?", type: "choice", options: ["Steady", "Throbbing", "Sharp", "Dull"] },
    { text: "Have you noticed vision changes, nausea, or dizziness?", type: "choice", options: ["Yes", "No"] }
  ],
  "Left eye": [
    { text: "Do you have eye pain, redness, or vision changes?", type: "choice", options: ["Pain", "Redness", "Vision changes", "None"] },
    { text: "Is there discharge, swelling, or sensitivity to light?", type: "choice", options: ["Discharge", "Swelling", "Light sensitivity", "None"] },
    { text: "Have you noticed blurred vision or spots/floaters?", type: "choice", options: ["Blurred vision", "Spots/floaters", "No"] }
  ],
  "Right eye": [
    { text: "Do you have eye pain, redness, or vision changes?", type: "choice", options: ["Pain", "Redness", "Vision changes", "None"] },
    { text: "Is there discharge, swelling, or sensitivity to light?", type: "choice", options: ["Discharge", "Swelling", "Light sensitivity", "None"] },
    { text: "Have you noticed blurred vision or spots/floaters?", type: "choice", options: ["Blurred vision", "Spots/floaters", "No"] }
  ],
  "Nose": [
    { text: "Do you have congestion, runny nose, or difficulty breathing?", type: "choice", options: ["Congestion", "Runny nose", "Difficulty breathing", "None"] },
    { text: "Is there pain, swelling, or discharge from the nose?", type: "choice", options: ["Pain", "Swelling", "Discharge", "None"] },
    { text: "Have you noticed loss of smell or frequent nosebleeds?", type: "choice", options: ["Loss of smell", "Nosebleeds", "No"] }
  ],
  "Left ear": [
    { text: "Do you have ear pain, ringing, or hearing loss?", type: "choice", options: ["Pain", "Ringing", "Hearing loss", "None"] },
    { text: "Is there discharge, itching, or swelling in the ear?", type: "choice", options: ["Discharge", "Itching", "Swelling", "None"] },
    { text: "Have you experienced dizziness or balance issues?", type: "choice", options: ["Dizziness", "Balance issues", "No"] }
  ],
  "Right ear": [
    { text: "Do you have ear pain, ringing, or hearing loss?", type: "choice", options: ["Pain", "Ringing", "Hearing loss", "None"] },
    { text: "Is there discharge, itching, or swelling in the ear?", type: "choice", options: ["Discharge", "Itching", "Swelling", "None"] },
    { text: "Have you experienced dizziness or balance issues?", type: "choice", options: ["Dizziness", "Balance issues", "No"] }
  ],
  "Mouth": [
    { text: "Do you have mouth pain, sores, or swelling?", type: "choice", options: ["Pain", "Sores", "Swelling", "None"] },
    { text: "Is there difficulty chewing, speaking, or tasting?", type: "choice", options: ["Chewing", "Speaking", "Tasting", "No"] },
    { text: "Have you noticed bad breath, bleeding gums, or tooth issues?", type: "choice", options: ["Bad breath", "Bleeding gums", "Tooth issues", "No"] }
  ],
  "Neck": [
    { text: "Is the discomfort in the front, back, or side of your neck?", type: "choice", options: ["Front", "Back", "Side", "All around"] },
    { text: "Does the pain get worse with movement or looking down?", type: "choice", options: ["Movement", "Looking down", "Holding still", "No"] },
    { text: "Do you feel stiffness, swelling, or numbness?", type: "choice", options: ["Stiffness", "Swelling", "Numbness", "None"] }
  ],
  "Left shoulder": [
    { text: "Is the pain sharp, aching, or more like pressure?", type: "choice", options: ["Sharp", "Aching", "Pressure", "None"] },
    { text: "Does it worsen with arm movement or lifting?", type: "choice", options: ["Arm movement", "Lifting", "Rest", "All of the above"] },
    { text: "Do you feel weakness, numbness, or radiating pain down the arm?", type: "choice", options: ["Weakness", "Numbness", "Radiating pain", "None"] }
  ],
  "Right shoulder": [
    { text: "Is the pain sharp, aching, or more like pressure?", type: "choice", options: ["Sharp", "Aching", "Pressure", "None"] },
    { text: "Does it worsen with arm movement or lifting?", type: "choice", options: ["Arm movement", "Lifting", "Rest", "All of the above"] },
    { text: "Do you feel weakness, numbness, or radiating pain down the arm?", type: "choice", options: ["Weakness", "Numbness", "Radiating pain", "None"] }
  ],
  "Chest": [
    { text: "Do you feel pain, tightness, or pressure?", type: "choice", options: ["Pain", "Tightness", "Pressure", "None"] },
    { text: "Is it worse with activity, breathing, or coughing?", type: "choice", options: ["Activity", "Breathing", "Coughing", "Rest"] },
    { text: "Any palpitations, dizziness, or shortness of breath?", type: "choice", options: ["Palpitations", "Dizziness", "Shortness of breath", "No"] }
  ],
  "Upper back": [
    { text: "Is the discomfort between your shoulder blades, around the spine, or both?", type: "choice", options: ["Between blades", "Around spine", "Both", "Neither"] },
    { text: "Does posture, deep breathing, or movement make it worse?", type: "choice", options: ["Posture", "Breathing", "Movement", "No"] },
    { text: "Do you feel tingling, stiffness, or soreness?", type: "choice", options: ["Tingling", "Stiffness", "Soreness", "None"] }
  ],
  "Abdomen": [
    { text: "Do you feel cramps, bloating, or sharp pain?", type: "choice", options: ["Cramps", "Bloating", "Sharp pain", "None"] },
    { text: "Is the discomfort related to eating, movement, or urination?", type: "choice", options: ["Eating", "Movement", "Urination", "No"] },
    { text: "Have you noticed nausea, vomiting, or digestion changes?", type: "choice", options: ["Nausea", "Vomiting", "Digestion changes", "No"] }
  ],
  "Lower back": [
    { text: "Is the pain sharp, dull, or aching?", type: "choice", options: ["Sharp", "Dull", "Aching", "None"] },
    { text: "Does bending, lifting, or standing make it worse?", type: "choice", options: ["Bending", "Lifting", "Standing", "No"] },
    { text: "Does the pain radiate into your legs?", type: "choice", options: ["Yes", "No", "Sometimes"] }
  ],
  "Pelvis": [
    { text: "Do you feel discomfort in the lower abdomen, groin, or hips?", type: "choice", options: ["Lower abdomen", "Groin", "Hips", "None"] },
    { text: "Is it worse when walking, standing, or sitting?", type: "choice", options: ["Walking", "Standing", "Sitting", "No"] },
    { text: "Any burning or changes when urinating?", type: "choice", options: ["Yes", "No", "Sometimes"] }
  ],
  "Left upper arm": [
    { text: "Do you feel pain, tightness, or weakness?", type: "choice", options: ["Pain", "Tightness", "Weakness", "None"] },
    { text: "Does it worsen with bending, reaching, or lifting?", type: "choice", options: ["Bending", "Reaching", "Lifting", "No"] }
  ],
  "Right upper arm": [
    { text: "Do you feel pain, tightness, or weakness?", type: "choice", options: ["Pain", "Tightness", "Weakness", "None"] },
    { text: "Does it worsen with bending, reaching, or lifting?", type: "choice", options: ["Bending", "Reaching", "Lifting", "No"] }
  ],
  "Left forearm": [
    { text: "Is there pain, numbness, or tingling?", type: "choice", options: ["Pain", "Numbness", "Tingling", "None"] },
    { text: "Does it worsen with gripping, twisting, or typing?", type: "choice", options: ["Gripping", "Twisting", "Typing", "No"] }
  ],
  "Right forearm": [
    { text: "Is there pain, numbness, or tingling?", type: "choice", options: ["Pain", "Numbness", "Tingling", "None"] },
    { text: "Does it worsen with gripping, twisting, or typing?", type: "choice", options: ["Gripping", "Twisting", "Typing", "No"] }
  ],
  "Left hand": [
    { text: "Is it aching, numb, or stiff?", type: "choice", options: ["Aching", "Numb", "Stiff", "None"] },
    { text: "Does it bother you when holding objects or using your fingers?", type: "choice", options: ["Holding objects", "Using fingers", "Rest", "No"] }
  ],
  "Right hand": [
    { text: "Is it aching, numb, or stiff?", type: "choice", options: ["Aching", "Numb", "Stiff", "None"] },
    { text: "Does it bother you when holding objects or using your fingers?", type: "choice", options: ["Holding objects", "Using fingers", "Rest", "No"] }
  ],
  "Left thigh": [
    { text: "Do you feel pain, cramping, or heaviness?", type: "choice", options: ["Pain", "Cramping", "Heaviness", "None"] },
    { text: "Is it worse when walking, climbing, or resting?", type: "choice", options: ["Walking", "Climbing", "Resting", "No"] }
  ],
  "Right thigh": [
    { text: "Do you feel pain, cramping, or heaviness?", type: "choice", options: ["Pain", "Cramping", "Heaviness", "None"] },
    { text: "Is it worse when walking, climbing, or resting?", type: "choice", options: ["Walking", "Climbing", "Resting", "No"] }
  ],
  "Left knee": [
    { text: "Is the pain above, behind, or around the kneecap?", type: "choice", options: ["Above", "Behind", "Around", "None"] },
    { text: "Does it hurt when bending, straightening, or bearing weight?", type: "choice", options: ["Bending", "Straightening", "Weight bearing", "No"] }
  ],
  "Right knee": [
    { text: "Is the pain above, behind, or around the kneecap?", type: "choice", options: ["Above", "Behind", "Around", "None"] },
    { text: "Does it hurt when bending, straightening, or bearing weight?", type: "choice", options: ["Bending", "Straightening", "Weight bearing", "No"] }
  ],
  "Left calf": [
    { text: "Is it aching, tight, or swollen?", type: "choice", options: ["Aching", "Tight", "Swollen", "None"] },
    { text: "Does it worsen with walking or standing?", type: "choice", options: ["Walking", "Standing", "Rest", "No"] }
  ],
  "Right calf": [
    { text: "Is it aching, tight, or swollen?", type: "choice", options: ["Aching", "Tight", "Swollen", "None"] },
    { text: "Does it worsen with walking or standing?", type: "choice", options: ["Walking", "Standing", "Rest", "No"] }
  ],
  "Left foot": [
    { text: "Is the discomfort in the heel, arch, or toes?", type: "choice", options: ["Heel", "Arch", "Toes", "All of the above"] },
    { text: "Does it hurt more when standing, walking, or at rest?", type: "choice", options: ["Standing", "Walking", "Rest", "No"] }
  ],
  "Right foot": [
    { text: "Is the discomfort in the heel, arch, or toes?", type: "choice", options: ["Heel", "Arch", "Toes", "All of the above"] },
    { text: "Does it hurt more when standing, walking, or at rest?", type: "choice", options: ["Standing", "Walking", "Rest", "No"] }
  ]
};
let selectedSymptomRegion = null;
let regionResponses = {};

function selectRegion(region) {
  if (selectedSymptomRegion === region) {
    cancelRegionSelection();
    return;
  }
  selectedSymptomRegion = region;
  regionResponses = {};
  document.getElementById('selectedRegionName').textContent = region;
  document.getElementById('regionQuestionPanel').hidden = false;
  renderRegionQuestions(region);
  document.querySelectorAll('.body-part').forEach(el => el.classList.toggle('active', el.dataset.region === region));
}

let isThumbDragging = false;
let thumbDragStart = 0;
let thumbStartTop = 0;

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function updateScrollThumb() {
  const chatArea = document.getElementById('chatArea');
  const track = document.getElementById('scrollTrack');
  const thumb = document.getElementById('scrollThumb');
  if (!chatArea || !track || !thumb) return;

  const visible = chatArea.clientHeight;
  const total = chatArea.scrollHeight;
  const trackHeight = track.clientHeight;
  const thumbHeight = Math.max((visible / total) * trackHeight, 36);
  const maxTop = trackHeight - thumbHeight;
  const ratio = total > visible ? chatArea.scrollTop / (total - visible) : 0;

  thumb.style.height = thumbHeight + 'px';
  thumb.style.top = ratio * maxTop + 'px';
  track.setAttribute('aria-valuenow', Math.round(ratio * 100));
}

function scrollToPosition(position) {
  const chatArea = document.getElementById('chatArea');
  const track = document.getElementById('scrollTrack');
  const thumb = document.getElementById('scrollThumb');
  if (!chatArea || !track || !thumb) return;

  const maxTop = track.clientHeight - thumb.clientHeight;
  const ratio = clamp(position / maxTop, 0, 1);
  chatArea.scrollTop = ratio * (chatArea.scrollHeight - chatArea.clientHeight);
}

function startThumbDrag(event) {
  const thumb = document.getElementById('scrollThumb');
  if (!thumb) return;
  isThumbDragging = true;
  thumbDragStart = event.clientY;
  thumbStartTop = thumb.offsetTop;
  document.body.style.userSelect = 'none';
}

function stopThumbDrag() {
  if (!isThumbDragging) return;
  isThumbDragging = false;
  document.body.style.userSelect = '';
}

function onThumbDrag(event) {
  if (!isThumbDragging) return;
  const track = document.getElementById('scrollTrack');
  const thumb = document.getElementById('scrollThumb');
  if (!track || !thumb) return;

  const delta = event.clientY - thumbDragStart;
  const newTop = clamp(thumbStartTop + delta, 0, track.clientHeight - thumb.clientHeight);
  scrollToPosition(newTop);
}

function initScrollBar() {
  const chatArea = document.getElementById('chatArea');
  const track = document.getElementById('scrollTrack');
  const thumb = document.getElementById('scrollThumb');
  if (!chatArea || !track || !thumb) return;

  updateScrollThumb();
  chatArea.addEventListener('scroll', updateScrollThumb);
  window.addEventListener('resize', updateScrollThumb);

  thumb.addEventListener('mousedown', startThumbDrag);
  track.addEventListener('click', event => {
    if (event.target === thumb) return;
    const rect = track.getBoundingClientRect();
    const clickY = event.clientY - rect.top;
    const targetTop = clickY - thumb.clientHeight / 2;
    scrollToPosition(targetTop);
  });

  document.addEventListener('mousemove', onThumbDrag);
  document.addEventListener('mouseup', stopThumbDrag);
  document.addEventListener('mouseleave', stopThumbDrag);
}

function scrollPage(offset) {
  const chatArea = document.getElementById('chatArea');
  if (!chatArea) return;
  chatArea.scrollBy({ top: offset, behavior: 'smooth' });
}

function cancelRegionSelection() {
  selectedSymptomRegion = null;
  regionResponses = {};
  document.getElementById('regionQuestionPanel').hidden = true;
  document.getElementById('regionQuestions').innerHTML = '';
  document.getElementById('submitRegionBtn').disabled = true;
  document.querySelectorAll('.body-part.active').forEach(el => el.classList.remove('active'));
}

function renderRegionQuestions(region) {
  const questions = SYMPTOM_REGIONS[region] || [];
  const container = document.getElementById('regionQuestions');
  container.innerHTML = '';

  questions.forEach((question, idx) => {
    const row = document.createElement('div');
    row.className = 'question-row';

    const label = document.createElement('div');
    label.className = 'question-label';
    label.textContent = question.text;
    row.appendChild(label);

    if (question.type === 'choice') {
      const choiceWrap = document.createElement('div');
      choiceWrap.className = 'question-choice';
      question.options.forEach(option => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.textContent = option;
        btn.className = regionResponses[idx] === option ? 'selected' : '';
        btn.onclick = () => {
          regionResponses[idx] = option;
          renderRegionQuestions(region);
          validateRegionButton();
        };
        choiceWrap.appendChild(btn);
      });
      row.appendChild(choiceWrap);
    } else {
      const textarea = document.createElement('textarea');
      textarea.className = 'question-input';
      textarea.id = `regionInput${idx}`;
      textarea.placeholder = question.placeholder || 'Add any details';
      textarea.value = regionResponses[idx] || '';
      textarea.oninput = () => {
        regionResponses[idx] = textarea.value.trim();
        validateRegionButton();
      };
      row.appendChild(textarea);
    }

    container.appendChild(row);
  });

  validateRegionButton();
}

function validateRegionButton() {
  const region = selectedSymptomRegion;
  if (!region) return;
  const questions = SYMPTOM_REGIONS[region] || [];
  const valid = questions.length > 0 && questions.every((_, idx) => {
    const answer = regionResponses[idx];
    return answer !== undefined && answer !== null && String(answer).trim() !== '';
  });
  document.getElementById('submitRegionBtn').disabled = !valid;
}

function submitRegionQuestions() {
  if (!selectedSymptomRegion) return;
  const questions = SYMPTOM_REGIONS[selectedSymptomRegion] || [];
  const summaryLines = [
    `Symptom map context:`,
    `Region: ${selectedSymptomRegion}`,
    `Details:`
  ];
  questions.forEach((question, idx) => {
    const answer = regionResponses[idx] || 'No response';
    summaryLines.push(`- ${question.text} ${answer}`);
  });
  const summary = summaryLines.join('\n');
  cancelRegionSelection();
  const input = document.getElementById('chatInput');
  input.value = summary;
  input.focus();
  sendMessage();
}

/* ── Init ── */
(async function init() {
  const res = await fetch('/api/conversations');
  conversations = await res.json();
  renderConvList();
  if (conversations.length) loadConversation(conversations[0].id);
  initScrollBar();
})();

/* ── Sidebar ── */
function toggleSidebar() {
  sidebarOpen = !sidebarOpen;
  document.getElementById('sidebar').classList.toggle('collapsed', !sidebarOpen);
  document.getElementById('overlay').classList.toggle('show', sidebarOpen && window.innerWidth <= 700);
}

/* ── Conversations CRUD ── */
async function createConversation() {
  const res = await fetch('/api/conversations', { method: 'POST' });
  const conv = await res.json();
  conversations.unshift(conv);
  renderConvList();
  loadConversation(conv.id);
  document.getElementById('chatInput').focus();
}

function loadConversation(id) {
  activeId = id;
  const conv = getConv(id);
  if (!conv) return;
  document.getElementById('topTitle').textContent = conv.title;
  renderMessages(conv.messages);
  markActive(id);
}

async function deleteConversation(id, e) {
  e.stopPropagation();
  await fetch('/api/conversations/' + id, { method: 'DELETE' });
  conversations = conversations.filter(c => c.id !== id);
  if (activeId === id) {
    activeId = null;
    document.getElementById('msgList').innerHTML = '';
    document.getElementById('topTitle').textContent = 'New Consultation';
    document.getElementById('welcome').style.display = '';
  }
  renderConvList();
}

async function renameConversation(id, e) {
  e.stopPropagation();
  const conv = getConv(id);
  if (!conv) return;
  const newTitle = prompt('Rename consultation', conv.title);
  if (!newTitle || !newTitle.trim() || newTitle.trim() === conv.title) return;
  const title = newTitle.trim();
  const res = await fetch('/api/conversations/' + id, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title })
  });
  if (!res.ok) return;
  conv.title = title;
  renderConvList();
  if (activeId === id) document.getElementById('topTitle').textContent = title;
}

function getConv(id) { return conversations.find(c => c.id === id); }
function filterConvs(q) { renderConvList(q); }

/* ── Render sidebar list ── */
function renderConvList(query = '') {
  const list = document.getElementById('convList');
  const q = query.toLowerCase();
  const filtered = conversations.filter(c => c.title.toLowerCase().includes(q));
  if (!filtered.length) {
    list.innerHTML = `<div class="no-convs">
      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>
      ${query ? 'No results found' : 'No consultations yet.<br>Start a new one above.'}
    </div>`;
    return;
  }
  list.innerHTML = filtered.map(c => `
    <div class="conv-item${c.id === activeId ? ' active' : ''}" onclick="loadConversation('${c.id}')">
      <div class="conv-dot"></div>
      <div class="conv-body">
        <div class="conv-title">${esc(c.title)}</div>
        <div class="conv-preview">${esc(getPreview(c))}</div>
      </div>
      <div class="conv-time">${timeAgo(c.created)}</div>
      <button class="conv-action" onclick="renameConversation('${c.id}', event)" title="Rename">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 20h9"/><path d="M16.5 3.5a2.121 2.121 0 0 1 3 3L8 18l-4 1 1-4L16.5 3.5z"/></svg>
      </button>
      <button class="conv-del" onclick="deleteConversation('${c.id}', event)" title="Delete">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/></svg>
      </button>
    </div>
  `).join('');
}

function markActive(id) {
  document.querySelectorAll('.conv-item').forEach(el => {
    el.classList.toggle('active', el.getAttribute('onclick')?.includes(id));
  });
}

function getPreview(c) {
  if (!c.messages.length) return 'No messages yet';
  const last = c.messages[c.messages.length - 1];
  return last.content.replace(/<[^>]+>/g, '').slice(0, 55) + '…';
}

/* ── Render messages ── */
function renderMessages(messages) {
  const list = document.getElementById('msgList');
  document.getElementById('welcome').style.display = messages.length ? 'none' : '';
  list.innerHTML = messages.map(m => buildBubble(m)).join('');
  scrollBottom();
  updateScrollThumb();
}

function buildBubble(m) {
  const isAI = m.role === 'assistant';
  return `
  <div class="msg-row${isAI ? '' : ' usr'}">
    <div class="msg-av ${isAI ? 'av-ai' : 'av-user'}">
      ${isAI ? '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M22 12h-4l-3 9L9 3l-3 9H2"/></svg>' : 'You'}
    </div>
    <div class="msg-body">
      <div class="msg-head">
        <span class="msg-who ${isAI ? 'who-ai' : 'who-user'}">${isAI ? 'Remy' : 'You'}</span>
        <span class="msg-ts">${m.time || ''}</span>
        ${isAI ? '<button class="copy-btn" onclick="copyMessage(this)" title="Copy message"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="9" y="9" width="13" height="13" rx="2" ry="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/></svg></button>' : ''}
      </div>
      <div class="bubble ${isAI ? 'b-ai' : 'b-user'}">${m.content}</div>
    </div>
  </div>`;
}

/* ── Format AI response (markdown-like → HTML) ── */
function formatAIResponse(text) {
  let h = esc(text);
  h = h.replace(/\(\*\*(.+?)\*\*\)/g, '<strong>$1</strong>');
  h = h.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  h = h.replace(/\*(.+?)\*/g,     '<em>$1</em>');
  h = h.replace(/`([^`]+)`/g,     '<code>$1</code>');
  h = h.replace(/^### (.+)$/gm,   '<h3>$1</h3>');
  h = h.replace(/^## (.+)$/gm,    '<h3>$1</h3>');
  h = h.replace(/\[URGENT:\s*(.+?)\]/g,  '<div class="urgent-block">🚨 <span>$1</span></div>');
  h = h.replace(/\[WARNING:\s*(.+?)\]/g, '<div class="warn-block">⚠️ <span>$1</span></div>');
  h = h.replace(/^[-•]\s+(.+)$/gm, '<li>$1</li>');
  h = h.replace(/(<li>[\s\S]*?<\/li>)/g, '<ul>$1</ul>');
  h = h.replace(/<\/ul>\s*<ul>/g, '');
  h = h.replace(/^\d+\.\s+(.+)$/gm, '<li>$1</li>');
  h = h.split(/\n\n+/).map(p => {
    p = p.trim();
    if (!p) return '';
    if (/^<(h3|ul|ol|li|div)/.test(p)) return p;
    return `<p>${p.replace(/\n/g, '<br>')}</p>`;
  }).join('');
  return h;
}

/* ── Send message (streaming) ── */
async function sendMessage() {
  if (busy) return;
  const inp = document.getElementById('chatInput');
  const text = inp.value.trim();
  if (!text) return;

  inp.value = '';
  inp.style.height = 'auto';

  if (!activeId) await createConversation();
  const conv = getConv(activeId);
  if (!conv) return;

  document.getElementById('welcome').style.display = 'none';

  const userMsg = { role: 'user', content: esc(text).replace(/\n/g,'<br>'), time: nowTime() };
  conv.messages.push(userMsg);
  appendBubble(userMsg);

  if (conv.messages.length === 1) {
    conv.title = text.slice(0, 46) + (text.length > 46 ? '…' : '');
    document.getElementById('topTitle').textContent = conv.title;
  }

  renderConvList();
  busy = true;
  document.getElementById('sendBtn').disabled = true;
  showTyping();

  let bubbleEl = null;
  let rawText = '';

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ conv_id: activeId, message: text })
    });

    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      throw new Error(data.error || `HTTP ${res.status}`);
    }

    hideTyping();

    /* Insert an empty AI bubble to stream text into */
    const placeholder = { role: 'assistant', content: '', time: nowTime() };
    appendBubble(placeholder);
    bubbleEl = document.getElementById('msgList').lastElementChild.querySelector('.bubble');

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });

      /* Process complete SSE lines */
      const lines = buf.split('\n');
      buf = lines.pop(); /* keep any incomplete trailing line */

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const payload = line.slice(6).trim();
        if (!payload || payload === '[DONE]') continue;
        let evt;
        try { evt = JSON.parse(payload); } catch { continue; }

        if (evt.t !== undefined) {
          /* Text chunk — format immediately so ** never shows as raw text */
          rawText += evt.t;
          bubbleEl.innerHTML = formatAIResponse(rawText);
          scrollBottom();
        } else if (evt.done) {
          /* Stream finished — final render with complete text */
          bubbleEl.innerHTML = formatAIResponse(rawText);
          if (evt.conv) {
            const idx = conversations.findIndex(c => c.id === activeId);
            if (idx !== -1) conversations[idx] = evt.conv;
          }
          if (evt.follow_up && evt.follow_up.length > 0) {
            /* Add follow-up questions */
            const followUpHtml = '<div class="follow-up-questions"><div class="follow-up-title">💡 Suggested follow-up questions:</div>' +
              evt.follow_up.map(q => `<button class="follow-up-btn" data-question="${esc(q)}">${esc(q)}</button>`).join('') +
              '</div>';
            bubbleEl.innerHTML += followUpHtml;
            
            /* Add click handlers for follow-up buttons */
            bubbleEl.querySelectorAll('.follow-up-btn').forEach(btn => {
              btn.addEventListener('click', function() {
                const question = this.getAttribute('data-question');
                if (question && !busy) {
                  // Visual feedback
                  this.style.opacity = '0.6';
                  this.textContent = 'Sending...';
                  this.disabled = true;
                  
                  useStarter(question);
                }
              });
            });
          }
          renderConvList();
          scrollBottom();
        } else if (evt.error) {
          throw new Error(evt.error);
        }
      }
    }
  } catch (err) {
    hideTyping();
    if (bubbleEl) {
      bubbleEl.innerHTML = `<p>Sorry, I ran into a connection issue. Please try again.</p><div class="warn-block">⚠️ ${esc(err.message)}</div>`;
    } else {
      appendBubble({
        role: 'assistant',
        content: `<p>Sorry, I ran into a connection issue. Please try again.</p><div class="warn-block">⚠️ ${esc(err.message)}</div>`,
        time: nowTime()
      });
    }
  }

  busy = false;
  document.getElementById('sendBtn').disabled = false;
  document.getElementById('chatInput').focus();
}

/* ── DOM helpers ── */
function appendBubble(m) {
  const list = document.getElementById('msgList');
  const div = document.createElement('div');
  div.innerHTML = buildBubble(m);
  list.appendChild(div.firstElementChild);
  scrollBottom();
  updateScrollThumb();
}
function showTyping() { document.getElementById('typingRow').style.display = 'flex'; scrollBottom(); updateScrollThumb(); }
function hideTyping()  { document.getElementById('typingRow').style.display = 'none'; }
function scrollBottom() { setTimeout(() => document.getElementById('anchor').scrollIntoView({ behavior:'smooth' }), 50); }

async function clearChat() {
  if (!activeId) return;
  if (!confirm('Clear all messages in this consultation?')) return;
  await fetch('/api/conversations/' + activeId + '/clear', { method: 'POST' });
  const conv = getConv(activeId);
  if (conv) { conv.messages = []; conv.title = 'New Consultation'; }
  document.getElementById('topTitle').textContent = 'New Consultation';
  renderMessages([]);
  renderConvList();
}

function useStarter(text) {
  if (busy) return; // Don't send if already processing
  document.getElementById('chatInput').value = text;
  sendMessage();
}

/* ── Textarea auto-grow ── */
function grow(el) { el.style.height = 'auto'; el.style.height = Math.min(el.scrollHeight, 160) + 'px'; }
function handleKey(e) { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); } }

/* ── Utils ── */
function esc(str) {
  return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
function nowTime() { return new Date().toLocaleTimeString([], { hour:'2-digit', minute:'2-digit' }); }
function timeAgo(ts) {
  const diff = (Date.now() - ts) / 1000;
  if (diff < 60)    return 'now';
  if (diff < 3600)  return Math.floor(diff/60) + 'm';
  if (diff < 86400) return Math.floor(diff/3600) + 'h';
  return Math.floor(diff/86400) + 'd';
}

function copyMessage(btn) {
  const bubble = btn.closest('.msg-body').querySelector('.bubble');
  if (!bubble) return;
  
  // Get plain text from the bubble, removing HTML tags
  const text = bubble.textContent || bubble.innerText || '';
  
  navigator.clipboard.writeText(text).then(() => {
    // Visual feedback
    const originalIcon = btn.innerHTML;
    btn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20,6 9,17 4,12"/></svg>';
    btn.style.background = 'rgba(34,197,94,.2)';
    btn.style.color = 'var(--green)';
    btn.style.borderColor = 'rgba(34,197,94,.3)';
    
    setTimeout(() => {
      btn.innerHTML = originalIcon;
      btn.style.background = '';
      btn.style.color = '';
      btn.style.borderColor = '';
    }, 2000);
  }).catch(err => {
    console.error('Failed to copy text: ', err);
    // Fallback for older browsers
    const textArea = document.createElement('textarea');
    textArea.value = text;
    document.body.appendChild(textArea);
    textArea.select();
    try {
      document.execCommand('copy');
      // Same visual feedback as above
      const originalIcon = btn.innerHTML;
      btn.innerHTML = '<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20,6 9,17 4,12"/></svg>';
      btn.style.background = 'rgba(34,197,94,.2)';
      btn.style.color = 'var(--green)';
      btn.style.borderColor = 'rgba(34,197,94,.3)';
      
      setTimeout(() => {
        btn.innerHTML = originalIcon;
        btn.style.background = '';
        btn.style.color = '';
        btn.style.borderColor = '';
      }, 2000);
    } catch (fallbackErr) {
      console.error('Fallback copy failed: ', fallbackErr);
    }
    document.body.removeChild(textArea);
  });
}

function openMedicalProfile() {
  const modal = document.getElementById('medicalModal');
  if (modal) {
    modal.style.display = 'flex';
    loadMedicalProfile();
  }
}

function closeMedicalProfile() {
  const modal = document.getElementById('medicalModal');
  if (modal) {
    modal.style.display = 'none';
  }
}

// Emergency Services Functions
function openEmergencyServices() {
  const modal = document.getElementById('emergencyModal');
  if (modal) {
    modal.style.display = 'flex';
    loadEmergencyServices();
  }
}

function closeEmergencyServices() {
  const modal = document.getElementById('emergencyModal');
  if (modal) {
    modal.style.display = 'none';
  }
}

async function loadEmergencyServices() {
  try {
    // Load countries
    const res = await fetch('/api/emergency/countries');
    const data = await res.json();
    const countrySelect = document.getElementById('emergencyCountry');
    countrySelect.innerHTML = '<option value="">Select Country</option>';
    data.countries.forEach(country => {
      const option = document.createElement('option');
      option.value = country;
      option.textContent = country;
      countrySelect.appendChild(option);
    });
  } catch (err) {
    console.error('Failed to load emergency countries:', err);
  }
}

async function onCountryChange() {
  const country = document.getElementById('emergencyCountry').value;
  const regionSelect = document.getElementById('emergencyRegion');
  const citySelect = document.getElementById('emergencyCity');
  
  regionSelect.innerHTML = '<option value="">Select Region First</option>';
  regionSelect.disabled = !country;
  
  citySelect.innerHTML = '<option value="">Select City (Optional)</option>';
  citySelect.disabled = true;
  
  if (country) {
    try {
      const res = await fetch(`/api/emergency/regions/${encodeURIComponent(country)}`);
      const data = await res.json();
      data.regions.forEach(region => {
        const option = document.createElement('option');
        option.value = region;
        option.textContent = region;
        regionSelect.appendChild(option);
      });
    } catch (err) {
      console.error('Failed to load regions:', err);
    }
  }
}

async function onRegionChange() {
  const country = document.getElementById('emergencyCountry').value;
  const region = document.getElementById('emergencyRegion').value;
  const citySelect = document.getElementById('emergencyCity');
  
  citySelect.innerHTML = '<option value="">Select City (Optional)</option>';
  citySelect.disabled = !region;
  
  if (region) {
    try {
      const res = await fetch(`/api/emergency/cities/${encodeURIComponent(country)}/${encodeURIComponent(region)}`);
      const data = await res.json();
      data.cities.forEach(city => {
        const option = document.createElement('option');
        option.value = city;
        option.textContent = city;
        citySelect.appendChild(option);
      });
      
      // Automatically load region numbers
      onCityChange();
    } catch (err) {
      console.error('Failed to load cities:', err);
    }
  }
}

async function onCityChange() {
  const country = document.getElementById('emergencyCountry').value;
  const region = document.getElementById('emergencyRegion').value;
  const city = document.getElementById('emergencyCity').value;
  
  if (!country || !region) {
    document.getElementById('emergencyNumbers').style.display = 'none';
    return;
  }
  
  try {
    let url = `/api/emergency/numbers/${encodeURIComponent(country)}/${encodeURIComponent(region)}`;
    if (city) {
      url = `/api/emergency/numbers/${encodeURIComponent(country)}/${encodeURIComponent(region)}/${encodeURIComponent(city)}`;
    }
    
    const res = await fetch(url);
    const data = await res.json();
    
    const numbersList = document.getElementById('numbersList');
    numbersList.innerHTML = '';
    
    // Handle both formats: array of numbers or object with cities
    let numbersData = city ? data.numbers : data.region_data;
    
    if (city) {
      // Display single city numbers
      if (Array.isArray(numbersData)) {
        numbersData.forEach(num => {
          const div = document.createElement('div');
          div.className = 'emergency-number';
          div.innerHTML = `
            <span class="number-label">${city} Ambulance:</span>
            <span class="number-value">${num}</span>
          `;
          numbersList.appendChild(div);
        });
      }
    } else {
      // Display all cities in region
      if (typeof numbersData === 'object' && !Array.isArray(numbersData)) {
        Object.entries(numbersData).forEach(([cityName, numbers]) => {
          if (Array.isArray(numbers)) {
            numbers.forEach(num => {
              const div = document.createElement('div');
              div.className = 'emergency-number';
              div.innerHTML = `
                <span class="number-label">${cityName} - Ambulance:</span>
                <span class="number-value">${num}</span>
              `;
              numbersList.appendChild(div);
            });
          }
        });
      }
    }
    
    document.getElementById('emergencyNumbers').style.display = 'block';
  } catch (err) {
    console.error('Failed to load emergency numbers:', err);
  }
}

function callEmergency() {
  const numberElements = document.querySelectorAll('.number-value');
  if (numberElements.length > 0) {
    const firstNumber =  numberElements[0].textContent.trim();
    // Some browsers may not support tel: protocol in all scenarios
    window.location.href = `tel:${firstNumber}`;
  }
}

async function loadMedicalProfile() {
  try {
    const res = await fetch('/api/medical');
    const profile = await res.json();
    
    // Populate form fields
    document.getElementById('consentGiven').checked = profile.consent_given || false;
    
    // Handle arrays - if only 'none', show empty; otherwise join with commas
    const formatArrayField = (arr) => {
      if (!arr || arr.length === 0) return '';
      if (arr.length === 1 && arr[0] === 'none') return 'none';
      return arr.filter(item => item !== 'none').join(', ');
    };
    
    // Handle text fields - show 'none' if that's the value, otherwise show the value
    const formatTextField = (value) => {
      return (value === 'none') ? 'none' : (value || '');
    };
    
    document.getElementById('allergies').value = formatArrayField(profile.allergies);
    document.getElementById('chronicConditions').value = formatArrayField(profile.chronic_conditions);
    document.getElementById('medications').value = formatArrayField(profile.medications);
    document.getElementById('deficiencies').value = formatArrayField(profile.deficiencies);
    document.getElementById('emergencyContacts').value = formatTextField(profile.emergency_contacts);
    document.getElementById('bloodType').value = formatTextField(profile.blood_type);
    document.getElementById('insuranceInfo').value = formatTextField(profile.insurance_info);
  } catch (err) {
    console.error('Failed to load medical profile:', err);
  }
}

async function saveMedicalProfile() {
  const data = {
    consent_given: document.getElementById('consentGiven').checked,
    allergies: document.getElementById('allergies').value.split(',').map(s => s.trim()).filter(s => s || s === 'none'),
    chronic_conditions: document.getElementById('chronicConditions').value.split(',').map(s => s.trim()).filter(s => s || s === 'none'),
    medications: document.getElementById('medications').value.split(',').map(s => s.trim()).filter(s => s || s === 'none'),
    deficiencies: document.getElementById('deficiencies').value.split(',').map(s => s.trim()).filter(s => s || s === 'none'),
    emergency_contacts: document.getElementById('emergencyContacts').value.trim(),
    blood_type: document.getElementById('bloodType').value.trim(),
    insurance_info: document.getElementById('insuranceInfo').value.trim()
  };
  
  // Convert empty arrays to ['none'] for clarity
  ['allergies', 'chronic_conditions', 'medications', 'deficiencies'].forEach(field => {
    if (data[field].length === 0) {
      data[field] = ['none'];
    }
  });
  
  // Handle text fields - set to 'none' if empty
  ['emergency_contacts', 'blood_type', 'insurance_info'].forEach(field => {
    if (!data[field] || data[field].trim() === '') {
      data[field] = 'none';
    }
  });
  
  try {
    const res = await fetch('/api/medical', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data)
    });
    
    const result = await res.json();
    
    if (res.ok) {
      alert(result.message || 'Medical profile saved successfully!');
      closeMedicalProfile();
    } else {
      alert('Error: ' + (result.error || 'Failed to save medical profile'));
    }
  } catch (err) {
    console.error('Failed to save medical profile:', err);
    alert('Failed to save medical profile');
  }
}

async function deleteMedicalProfile() {
  if (!confirm('Are you sure you want to delete your medical profile? This action cannot be undone.')) {
    return;
  }
  
  try {
    const res = await fetch('/api/medical', { method: 'DELETE' });
    const result = await res.json();
    
    if (res.ok) {
      alert(result.message || 'Medical profile deleted successfully!');
      closeMedicalProfile();
      // Clear form
      document.querySelectorAll('#medicalModal input, #medicalModal textarea').forEach(el => el.value = '');
      document.getElementById('consentGiven').checked = false;
    } else {
      alert('Error: ' + (result.error || 'Failed to delete medical profile'));
    }
  } catch (err) {
    console.error('Failed to delete medical profile:', err);
    alert('Failed to delete medical profile');
  }
}
</script>

<!-- Medical Profile Modal -->
<div id="medicalModal" class="modal-overlay" style="display: none;">
  <div class="modal-content">
    <div class="modal-header">
      <h3>Medical Profile</h3>
      <button class="modal-close" onclick="closeMedicalProfile()">×</button>
    </div>
    <div class="modal-body">
      <div class="consent-section">
        <label class="consent-label">
          <input type="checkbox" id="consentGiven">
          <span>I consent to Remy using my medical information for personalized safety and health guidance</span>
        </label>
        <p class="consent-note">Your medical data is stored securely and only used to provide better safety recommendations. You can update or delete it at any time.</p>
      </div>
      
      <div class="form-group">
        <label for="allergies">Allergies (comma-separated):</label>
        <input type="text" id="allergies" placeholder="e.g., peanuts, penicillin, latex (or 'none')">
      </div>
      
      <div class="form-group">
        <label for="chronicConditions">Chronic Conditions (comma-separated):</label>
        <input type="text" id="chronicConditions" placeholder="e.g., asthma, diabetes, hypertension (or 'none')">
      </div>
      
      <div class="form-group">
        <label for="medications">Current Medications (comma-separated):</label>
        <input type="text" id="medications" placeholder="e.g., metformin, lisinopril (or 'none')">
      </div>
      
      <div class="form-group">
        <label for="deficiencies">Known Deficiencies (comma-separated):</label>
        <input type="text" id="deficiencies" placeholder="e.g., vitamin D, iron (or 'none')">
      </div>
      
      <div class="form-group">
        <label for="emergencyContacts">Emergency Contacts:</label>
        <textarea id="emergencyContacts" placeholder="Name and phone number of emergency contacts (or 'none')"></textarea>
      </div>
      
      <div class="form-group">
        <label for="bloodType">Blood Type:</label>
        <input type="text" id="bloodType" placeholder="e.g., O+, A-, AB+ (or 'none')">
      </div>
      
      <div class="form-group">
        <label for="insuranceInfo">Insurance Information:</label>
        <textarea id="insuranceInfo" placeholder="Insurance provider, policy number, etc. (or 'none')"></textarea>
      </div>
    </div>
    <div class="modal-footer">
      <button class="btn-secondary" onclick="deleteMedicalProfile()">Delete Profile</button>
      <div>
        <button class="btn-secondary" onclick="closeMedicalProfile()">Cancel</button>
        <button class="btn-primary" onclick="saveMedicalProfile()">Save Profile</button>
      </div>
    </div>
  </div>
</div>

<!-- Emergency Services Modal -->
<div id="emergencyModal" class="modal-overlay" style="display: none;">
  <div class="modal-content">
    <div class="modal-header">
      <h3>Emergency Ambulance Services</h3>
      <button class="modal-close" onclick="closeEmergencyServices()">×</button>
    </div>
    <div class="modal-body">
      <div class="emergency-notice">
        <div class="emergency-alert">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
          <strong>Emergency Notice:</strong> If you are experiencing a medical emergency, call emergency services immediately. This tool helps you find the correct ambulance number for your location.
        </div>
      </div>
      
      <div class="location-selector">
        <div class="form-group">
          <label for="emergencyCountry">Country:</label>
          <select id="emergencyCountry" onchange="onCountryChange()">
            <option value="">Select Country</option>
          </select>
        </div>
        
        <div class="form-group">
          <label for="emergencyRegion">Region/State/Province:</label>
          <select id="emergencyRegion" onchange="onRegionChange()" disabled>
            <option value="">Select Region First</option>
          </select>
        </div>
        
        <div class="form-group">
          <label for="emergencyCity">City (Optional - shows all if not selected):</label>
          <select id="emergencyCity" onchange="onCityChange()" disabled>
            <option value="">Select City (Optional)</option>
          </select>
        </div>
      </div>
      
      <div id="emergencyNumbers" class="emergency-numbers" style="display: none;">
        <h4>Emergency Ambulance Numbers:</h4>
        <div id="numbersList" class="numbers-list">
          <!-- Numbers will be populated here -->
        </div>
        <div class="emergency-actions">
          <button class="btn-primary emergency-call-btn" onclick="callEmergency()" style="display: none;">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"/></svg>
            Call Emergency Services
          </button>
          <p class="emergency-disclaimer">
            <strong>Important:</strong> Only call emergency services in genuine emergencies. Misuse can prevent real emergencies from getting help.
          </p>
        </div>
      </div>
    </div>
    <div class="modal-footer">
      <button class="btn-secondary" onclick="closeEmergencyServices()">Close</button>
    </div>
  </div>
</div>

<style>
.modal-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 1000;
}

.modal-content {
  background: var(--navy-900);
  border: 1px solid var(--border-subtle);
  border-radius: var(--r-lg);
  max-width: 600px;
  width: 90%;
  max-height: 80vh;
  overflow-y: auto;
}

.modal-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px 24px;
  border-bottom: 1px solid var(--border-faint);
}

.modal-header h3 {
  margin: 0;
  color: var(--text-primary);
  font-size: 18px;
  font-weight: 600;
}

.modal-close {
  background: none;
  border: none;
  color: var(--text-muted);
  font-size: 24px;
  cursor: pointer;
  padding: 0;
  width: 30px;
  height: 30px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 4px;
}

.modal-close:hover {
  background: var(--navy-800);
  color: var(--text-primary);
}

.modal-body {
  padding: 24px;
}

.consent-section {
  margin-bottom: 24px;
  padding: 16px;
  background: var(--navy-850);
  border: 1px solid var(--border-faint);
  border-radius: var(--r);
}

.consent-label {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  cursor: pointer;
  font-weight: 500;
  color: var(--text-primary);
}

.consent-label input[type="checkbox"] {
  margin-top: 2px;
  width: 16px;
  height: 16px;
}

.consent-note {
  margin: 12px 0 0 28px;
  font-size: 12px;
  color: var(--text-secondary);
  line-height: 1.5;
}

.form-group {
  margin-bottom: 20px;
}

.form-group label {
  display: block;
  margin-bottom: 6px;
  font-weight: 500;
  color: var(--text-primary);
  font-size: 14px;
}

.form-group input,
.form-group textarea {
  width: 100%;
  padding: 10px 12px;
  border: 1px solid var(--border-faint);
  border-radius: var(--r-sm);
  background: var(--navy-850);
  color: var(--text-primary);
  font-size: 14px;
}

.form-group input:focus,
.form-group textarea:focus {
  outline: none;
  border-color: var(--accent-mid);
  box-shadow: 0 0 0 2px rgba(47, 158, 255, 0.1);
}

.form-group textarea {
  resize: vertical;
  min-height: 60px;
}

.modal-footer {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 20px 24px;
  border-top: 1px solid var(--border-faint);
}

.btn-primary,
.btn-secondary {
  padding: 10px 16px;
  border-radius: var(--r-sm);
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
}

.btn-primary {
  background: var(--accent);
  color: white;
  border: 1px solid var(--accent);
}

.btn-primary:hover {
  background: var(--accent-glow);
}

.btn-secondary {
  background: var(--navy-800);
  color: var(--text-primary);
  border: 1px solid var(--border-subtle);
}

.btn-secondary:hover {
  background: var(--navy-700);
  border-color: var(--border-mid);
}

/* Emergency Services Styles */
.emergency-btn {
  background: linear-gradient(135deg, #dc2626, #ef4444) !important;
  border: 1px solid #dc2626 !important;
}

.emergency-btn:hover {
  background: linear-gradient(135deg, #b91c1c, #dc2626) !important;
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(220, 38, 38, 0.3);
}

.emergency-notice {
  margin-bottom: 24px;
}

.emergency-alert {
  display: flex;
  align-items: flex-start;
  gap: 12px;
  padding: 16px;
  background: linear-gradient(135deg, rgba(220, 38, 38, 0.1), rgba(239, 68, 68, 0.1));
  border: 1px solid rgba(220, 38, 38, 0.3);
  border-radius: var(--r);
  color: #fca5a5;
}

.emergency-alert svg {
  color: #ef4444;
  flex-shrink: 0;
  margin-top: 2px;
}

.location-selector {
  margin-bottom: 24px;
}

.form-group select {
  width: 100%;
  padding: 10px 12px;
  border: 1px solid var(--border-faint);
  border-radius: var(--r-sm);
  background: var(--navy-850);
  color: var(--text-primary);
  font-size: 14px;
  cursor: pointer;
}

.form-group select:focus {
  outline: none;
  border-color: var(--accent-mid);
  box-shadow: 0 0 0 2px rgba(47, 158, 255, 0.1);
}

.form-group select:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.emergency-numbers {
  border-top: 1px solid var(--border-faint);
  padding-top: 24px;
}

.emergency-numbers h4 {
  margin: 0 0 16px 0;
  color: var(--text-primary);
  font-size: 16px;
  font-weight: 600;
}

.numbers-list {
  margin-bottom: 20px;
}

.emergency-number {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  margin-bottom: 8px;
  background: var(--navy-850);
  border: 1px solid var(--border-faint);
  border-radius: var(--r-sm);
  transition: all 0.2s;
}

.emergency-number:hover {
  background: var(--navy-800);
  border-color: var(--accent-mid);
}

.emergency-number:last-child {
  margin-bottom: 0;
}

.number-label {
  font-weight: 500;
  color: var(--text-primary);
}

.number-value {
  font-family: 'Courier New', monospace;
  font-size: 16px;
  font-weight: 600;
  color: var(--accent);
  letter-spacing: 1px;
}

.emergency-actions {
  text-align: center;
}

.emergency-call-btn {
  margin-bottom: 16px;
  background: linear-gradient(135deg, #dc2626, #ef4444) !important;
  border: 1px solid #dc2626 !important;
  font-size: 16px;
  padding: 12px 24px;
  width: 100%;
}

.emergency-call-btn:hover {
  background: linear-gradient(135deg, #b91c1c, #dc2626) !important;
  transform: translateY(-1px);
  box-shadow: 0 4px 12px rgba(220, 38, 38, 0.3);
}

.emergency-disclaimer {
  font-size: 12px;
  color: var(--text-secondary);
  line-height: 1.5;
  margin: 0;
  padding: 12px;
  background: var(--navy-850);
  border-radius: var(--r-sm);
  border-left: 3px solid #ef4444;
}

.emergency-disclaimer strong {
  color: #fca5a5;
}

/* Responsive adjustments */
@media (max-width: 768px) {
  .emergency-number {
    flex-direction: column;
    align-items: flex-start;
    gap: 8px;
  }
  
  .number-value {
    font-size: 18px;
  }
}
</style>

</body>
</html>"""

AUTH_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{{ page_title }} — MedAI</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&family=DM+Serif+Display:ital@0;1&display=swap" rel="stylesheet" />
  <style>
    :root {
      --navy-950: #040c18;
      --navy-900: #071426;
      --navy-700: #172e52;
      --accent: #2f9eff;
      --accent-glow: #5ab4ff;
      --text-primary: #deeaf6;
      --text-secondary: #6b90b4;
      --red: #ef4444;
      --green: #22c55e;
    }
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    html, body { height: 100%; background: var(--navy-950); color: var(--text-primary); font-family: 'DM Sans', sans-serif; }
    body { display: flex; align-items: center; justify-content: center; padding: 20px; }
    .auth-container { width: 100%; max-width: 400px; }
    .auth-header { text-align: center; margin-bottom: 40px; }
    .auth-logo { font-size: 32px; font-weight: 600; margin-bottom: 8px; background: linear-gradient(135deg, var(--accent), #1668b0); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    .auth-subtitle { font-size: 14px; color: var(--text-secondary); }
    .auth-form { display: flex; flex-direction: column; gap: 16px; }
    .form-group { display: flex; flex-direction: column; gap: 6px; }
    .form-label { font-size: 13px; font-weight: 500; color: var(--text-secondary); }
    .form-input { padding: 11px 14px; border: 1px solid rgba(255,255,255,0.1); background: rgba(255,255,255,0.04); border-radius: 8px; color: var(--text-primary); font-size: 14px; transition: all 0.2s; }
    .form-input:focus { outline: none; border-color: var(--accent); background: rgba(255,255,255,0.06); }
    .form-input::placeholder { color: var(--text-secondary); }
    .form-submit { padding: 11px 16px; background: linear-gradient(135deg, var(--accent), #1668b0); color: white; border: none; border-radius: 8px; font-weight: 600; cursor: pointer; font-size: 14px; transition: all 0.2s; }
    .form-submit:hover { box-shadow: 0 0 22px rgba(47,158,255,0.3); }
    .form-submit:active { transform: scale(0.98); }
    .auth-message { padding: 11px 14px; border-radius: 8px; font-size: 13px; text-align: center; }
    .auth-message.error { background: rgba(239,68,68,0.1); color: var(--red); border: 1px solid rgba(239,68,68,0.2); }
    .auth-message.success { background: rgba(34,197,94,0.1); color: var(--green); border: 1px solid rgba(34,197,94,0.2); }
    .auth-footer { text-align: center; margin-top: 24px; font-size: 14px; }
    .auth-footer a { color: var(--accent); text-decoration: none; }
    .auth-footer a:hover { text-decoration: underline; }
  </style>
</head>
<body>
  <div class="auth-container">
    <div class="auth-header">
      <div class="auth-logo">MedAI</div>
      <div class="auth-subtitle">{{ page_subtitle }}</div>
    </div>
    {% if message %}
      <div class="auth-message {{ message_type }}">{{ message }}</div>
    {% endif %}
    <form method="POST" class="auth-form">
      <div class="form-group">
        <label class="form-label">Email Address</label>
        <input type="email" name="email" class="form-input" placeholder="you@example.com" required autofocus />
      </div>
      <div class="form-group">
        <label class="form-label">Password</label>
        <input type="password" name="password" class="form-input" placeholder="••••••••" required />
      </div>
      {% if is_signup %}
        <div class="form-group">
          <label class="form-label">Confirm Password</label>
          <input type="password" name="confirm_password" class="form-input" placeholder="••••••••" required />
        </div>
      {% endif %}
      <button type="submit" class="form-submit">{{ button_text }}</button>
    </form>
    <div class="auth-footer">
      {% if is_signup %}
        Already have an account? <a href="/signin">Sign In</a>
      {% else %}
        Don't have an account? <a href="/signup">Sign Up</a>
      {% endif %}
    </div>
  </div>
</body>
</html>"""


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    if "user_email" not in session:
        return redirect(url_for("signin"))
    return render_template_string(HTML_TEMPLATE)


@app.route("/signup", methods=["GET", "POST"])
def signup():
    message = ""
    message_type = ""
    
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        confirm_password = request.form.get("confirm_password", "").strip()
        
        if password != confirm_password:
            message = "Passwords do not match"
            message_type = "error"
        else:
            success, msg = register_user(email, password)
            if success:
                message = msg
                message_type = "success"
                # Auto-login after successful registration
                session["user_email"] = email
                return redirect(url_for("index"))
            else:
                message = msg
                message_type = "error"
    
    return render_template_string(
        AUTH_TEMPLATE,
        is_signup=True,
        page_title="Sign Up",
        page_subtitle="Create your MedAI account",
        button_text="Sign Up",
        message=message,
        message_type=message_type
    )


@app.route("/signin", methods=["GET", "POST"])
def signin():
    message = ""
    message_type = ""
    
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "").strip()
        
        success, result = login_user(email, password)
        if success:
            session["user_email"] = result
            return redirect(url_for("index"))
        else:
            message = result
            message_type = "error"
    
    return render_template_string(
        AUTH_TEMPLATE,
        is_signup=False,
        page_title="Sign In",
        page_subtitle="Welcome back to MedAI",
        button_text="Sign In",
        message=message,
        message_type=message_type
    )


@app.route("/signout")
def signout():
    session.clear()
    return redirect(url_for("signin"))


def _require_login(f):
    """Decorator to require user login for API endpoints"""
    def wrapper(*args, **kwargs):
        if "user_email" not in session:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper


@app.route("/api/conversations", methods=["GET"])
@_require_login
def get_conversations():
    _ensure_loaded()
    return jsonify(_convs)


@app.route("/api/conversations", methods=["POST"])
@_require_login
def new_conversation():
    _ensure_loaded()
    conv = {
        "id": "conv_" + uuid.uuid4().hex,
        "title": "New Consultation",
        "messages": [],
        "created": int(time.time() * 1000),
    }
    _convs.insert(0, conv)
    _flush()
    return jsonify(conv), 201


@app.route("/api/conversations/<conv_id>", methods=["DELETE"])
@_require_login
def delete_conversation(conv_id):
    _ensure_loaded()
    _convs[:] = [c for c in _convs if c["id"] != conv_id]
    _flush()
    return jsonify({"ok": True})


@app.route("/api/conversations/<conv_id>", methods=["PATCH"])
@_require_login
def rename_conversation(conv_id):
    body = request.get_json(force=True)
    title = body.get("title", "").strip()
    if not title:
        return jsonify({"error": "Title is required"}), 400

    _ensure_loaded()
    for c in _convs:
        if c["id"] == conv_id:
            c["title"] = title
            _flush()
            return jsonify({"ok": True, "id": conv_id, "title": title})

    return jsonify({"error": "Conversation not found"}), 404


@app.route("/api/conversations/<conv_id>/clear", methods=["POST"])
@_require_login
def clear_conversation(conv_id):
    _ensure_loaded()
    for c in _convs:
        if c["id"] == conv_id:
            c["messages"] = []
            c["title"] = "New Consultation"
            break
    _flush()
    return jsonify({"ok": True})


@app.route("/api/medical", methods=["GET"])
@_require_login
def get_medical():
    """Get user's medical profile"""
    user_email = session.get('user_email')
    if not user_email:
        return jsonify({"error": "Not logged in"}), 401
    
    profile = get_medical_profile(user_email)
    return jsonify(profile)


@app.route("/api/medical", methods=["POST"])
@_require_login
def update_medical():
    """Update user's medical profile"""
    user_email = session.get('user_email')
    if not user_email:
        return jsonify({"error": "Not logged in"}), 401
    
    data = request.get_json(force=True)
    success, message = update_medical_profile(user_email, data)
    
    if success:
        return jsonify({"ok": True, "message": message})
    else:
        return jsonify({"error": message}), 400


@app.route("/api/medical", methods=["DELETE"])
@_require_login
def delete_medical():
    """Delete user's medical profile"""
    user_email = session.get('user_email')
    if not user_email:
        return jsonify({"error": "Not logged in"}), 401
    
    success, message = delete_medical_profile(user_email)
    
    if success:
        return jsonify({"ok": True, "message": message})
    else:
        return jsonify({"error": message}), 400


@app.route("/api/emergency/countries", methods=["GET"])
@_require_login
def get_emergency_countries():
    """Get list of available countries for emergency services"""
    try:
        countries = get_available_countries()
        return jsonify({"countries": countries})
    except Exception as e:
        return jsonify({"error": f"Failed to load countries: {str(e)}"}), 500

@app.route("/api/emergency/regions/<country>", methods=["GET"])
@_require_login
def get_emergency_regions(country):
    """Get list of available regions for a country"""
    try:
        regions = get_available_regions(country)
        return jsonify({"regions": regions})
    except Exception as e:
        return jsonify({"error": f"Failed to load regions: {str(e)}"}), 500

@app.route("/api/emergency/cities/<country>/<region>", methods=["GET"])
@_require_login
def get_emergency_cities(country, region):
    """Get list of available cities for a country and region"""
    try:
        cities = get_available_cities(country, region)
        return jsonify({"cities": cities})
    except Exception as e:
        return jsonify({"error": f"Failed to load cities: {str(e)}"}), 500

@app.route("/api/emergency/numbers/<country>/<region>/<city>", methods=["GET"])
@_require_login
def get_emergency_numbers(country, region, city):
    """Get emergency numbers for a specific location"""
    try:
        numbers = get_emergency_services(country, region, city)
        return jsonify({"numbers": numbers})
    except Exception as e:
        return jsonify({"error": f"Failed to load emergency numbers: {str(e)}"}), 500

@app.route("/api/emergency/numbers/<country>/<region>", methods=["GET"])
@_require_login
def get_emergency_numbers_region(country, region):
    """Get emergency numbers for a region (all cities)"""
    try:
        region_data = get_emergency_services(country, region)
        return jsonify({"region_data": region_data})
    except Exception as e:
        return jsonify({"error": f"Failed to load emergency numbers: {str(e)}"}), 500


@app.route("/api/chat", methods=["POST"])
@_require_login
def chat():
    try:
        body = request.get_json(force=True)
        if not body:
            return jsonify({"error": "Invalid request body"}), 400

        conv_id: str = body.get("conv_id", "")
        user_text: str = body.get("message", "").strip()

        if not user_text:
            return jsonify({"error": "Empty message"}), 400

        _ensure_loaded()
        conv = next((c for c in _convs if c["id"] == conv_id), None)
        if conv is None:
            return jsonify({"error": "Conversation not found"}), 404

        # Build API history from existing messages
        user_email = session.get("user_email")
        medical_context = ""
        try:
            if user_email:
                medical_profile = get_medical_profile(user_email)
                if medical_profile and medical_profile.get("consent_given"):
                    medical_context = (
                        f"\n\nUSER MEDICAL PROFILE (use only with explicit consent):\n"
                        f"{json.dumps(medical_profile, indent=2)}"
                    )
        except Exception:
            pass  # Medical profile is optional — never block chat for this

        enhanced_prompt = SYSTEM_PROMPT + medical_context
        api_messages = [{"role": "system", "content": enhanced_prompt}]
        for m in conv["messages"]:
            clean = _RE_TAGS.sub("", m["content"].replace("<br>", "\n"))
            api_messages.append({"role": m["role"], "content": clean})
        api_messages.append({"role": "user", "content": user_text})

        timestamp = datetime.now().strftime("%I:%M %p")

    except Exception as exc:
        return jsonify({"error": f"Request setup failed: {str(exc)}"}), 500

    def generate():
        chunks: list[str] = []

        # ── Stream from Cohere ────────────────────────────────────────────────
        try:
            resp = _http.post(
                COHERE_URL,
                json={"model": COHERE_MODEL, "messages": api_messages, "stream": True},
                stream=True,
                timeout=60,
            )
            resp.raise_for_status()

            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                if event.get("type") == "content-delta":
                    chunk = (
                        event.get("delta", {})
                             .get("message", {})
                             .get("content", {})
                             .get("text", "")
                    )
                    if chunk:
                        chunks.append(chunk)
                        yield f'data: {json.dumps({"t": chunk})}\n\n'

        except requests.HTTPError as exc:
            try:
                msg = exc.response.json().get("message", str(exc))
            except Exception:
                msg = str(exc)
            yield f'data: {json.dumps({"error": msg})}\n\n'
            return
        except Exception as exc:
            yield f'data: {json.dumps({"error": str(exc)})}\n\n'
            return

        # ── Persist and send done event ───────────────────────────────────────
        try:
            raw_reply = "".join(chunks)
            html_reply = format_reply(raw_reply) if raw_reply else "<p>No response received.</p>"

            conv["messages"].append({
                "role": "user",
                "content": user_text.replace("\n", "<br>"),
                "time": timestamp,
            })
            if len(conv["messages"]) == 1:
                conv["title"] = user_text[:46] + ("…" if len(user_text) > 46 else "")
            conv["messages"].append({
                "role": "assistant",
                "content": html_reply,
                "time": timestamp,
            })

            try:
                _flush()
            except Exception as flush_err:
                print(f"[MedAI] Warning: could not save conversation: {flush_err}")

            yield f'data: {json.dumps({"done": True, "conv": conv})}\n\n'

        except Exception as exc:
            yield f'data: {json.dumps({"error": f"Failed to save response: {str(exc)}"}  )}\n\n'

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )




# ── Markdown-like → HTML formatter (uses pre-compiled patterns) ───────────────

_RE_BLOCK_START = re.compile(r"^<(h3|ul|ol|li|div)")

def format_reply(text: str) -> str:
    h = html_mod.escape(text)

    h = _RE_BOLD.sub(r"<strong>\1</strong>", h)
    h = _RE_ITALIC.sub(r"<em>\1</em>", h)
    h = _RE_CODE.sub(r"<code>\1</code>", h)
    h = _RE_H3.sub(r"<h3>\1</h3>", h)
    h = _RE_H2.sub(r"<h3>\1</h3>", h)
    h = _RE_URGENT.sub(r'<div class="urgent-block">🚨 <span>\1</span></div>', h)
    h = _RE_WARNING.sub(r'<div class="warn-block">⚠️ <span>\1</span></div>', h)

    # Convert bullet/numbered markers to <li> tags
    h = _RE_BULLET.sub(r"<li>\1</li>", h)
    h = _RE_NUMLIST.sub(r"<li>\1</li>", h)

    # Wrap consecutive <li> lines in a single <ul> — line-by-line loop avoids
    # the RecursionError that (<li>[\s\S]*?</li>) caused on long responses
    lines = h.splitlines()
    wrapped: list[str] = []
    in_list = False
    for line in lines:
        if line.strip().startswith("<li>"):
            if not in_list:
                wrapped.append("<ul>")
                in_list = True
            wrapped.append(line)
        else:
            if in_list:
                wrapped.append("</ul>")
                in_list = False
            wrapped.append(line)
    if in_list:
        wrapped.append("</ul>")
    h = "\n".join(wrapped)

    result = []
    for p in re.split(r"\n\n+", h):
        p = p.strip()
        if not p:
            continue
        if _RE_BLOCK_START.match(p):
            result.append(p)
        else:
            result.append(f"<p>{p.replace(chr(10), '<br>')}</p>")
    return "".join(result)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Remy starting — open http://127.0.0.1:5000")
    app.run(debug=True, port=5000, threaded=True)
