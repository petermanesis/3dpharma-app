"""
Drug Service - Core business logic for drug queries
Adapted from the Streamlit app's ComprehensiveDrugQuery class
"""

import os
import re
import json
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Any

try:
    import ijson
    IJSON_AVAILABLE = True
except ImportError:
    IJSON_AVAILABLE = False

from bs4 import BeautifulSoup

from app.services.severity_classifier import classify_severity_simple, get_severity_emoji


def normalize_route(route: str) -> str:
    """Normalize route names to consistent title case."""
    if not route:
        return route
    # Handle all-caps routes
    if route.isupper():
        route = route.title()
    # Common normalizations
    route = route.replace('ORAL', 'Oral').replace('oral', 'Oral')
    route = route.replace('(INHALATION)', '(inhalation)')
    return route


class DrugService:
    """Query interface for comprehensive database with OpenFDA fallback"""

    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        """Singleton pattern to avoid reloading the database"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    # Google Drive data source
    GDRIVE_FILE_ID = '12o_cdObA01lxXJMY8LjCqlPVrXF56bZF'
    GDRIVE_URL = f'https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}'
    
    def _load_from_gdrive(self) -> dict:
        """Load database directly from Google Drive (no local file needed)."""
        try:
            import httpx
        except ImportError:
            print("Warning: 'httpx' library not installed. Cannot connect to Google Drive.")
            print("Install with: pip install httpx")
            return None
        
        print(f"Connecting to Google Drive database...")
        print(f"Loading ~200MB file directly from cloud (this may take a minute)...")
        try:
            with httpx.Client(follow_redirects=True, timeout=300.0) as client:
                # Google Drive requires confirm=t for large files (virus scan bypass)
                download_url = f"{self.GDRIVE_URL}&confirm=t"
                response = client.get(download_url)
                
                # If response is not JSON, it's likely an HTML warning page
                content_start = response.content[:2000]
                if b'"drugs"' not in content_start and (b'<html' in content_start or b'<!DOCTYPE' in content_start or len(response.content) < 1000):
                    # Try extracting uuid/confirm token from the HTML page
                    uuid_match = re.search(r'uuid=([^&"\']+)', response.text)
                    confirm_match = re.search(r'confirm=([^&"\']+)', response.text)
                    
                    if uuid_match:
                        download_url = f"https://drive.usercontent.google.com/download?id={self.GDRIVE_FILE_ID}&export=download&confirm=t&uuid={uuid_match.group(1)}"
                        response = client.get(download_url)
                    elif confirm_match and confirm_match.group(1) != 't':
                        download_url = f"{self.GDRIVE_URL}&confirm={confirm_match.group(1)}"
                        response = client.get(download_url)
                    else:
                        # Try the usercontent domain directly
                        download_url = f"https://drive.usercontent.google.com/download?id={self.GDRIVE_FILE_ID}&export=download&confirm=t"
                        response = client.get(download_url)
                
                if not response.content:
                    print("Failed to load from Google Drive: empty response")
                    return None
                
                # Parse JSON directly in memory
                data = json.loads(response.content.decode('utf-8'))
                print(f"Successfully loaded {len(data.get('drugs', []))} drugs from Google Drive")
                return data
        except Exception as e:
            print(f"Failed to load from Google Drive: {e}")
            return None
    
    def __init__(
        self,
        db_file: str = 'comprehensive_drug_database_compact.json',
        openfda_file: str = 'OpenFDAfull.json',
    ):
        # Avoid re-initialization
        if DrugService._initialized:
            return
        
        # Try to find database in multiple locations
        db_paths = [
            db_file,
            os.path.join('data', db_file),
            os.path.join('..', 'drug-app', db_file),
            os.path.join('..', 'data', db_file),
            'comprehensive_drug_database.json',
            os.path.join('data', 'comprehensive_drug_database.json'),
        ]
        
        self.db_file = None
        for path in db_paths:
            if os.path.exists(path):
                self.db_file = path
                break
        
        self.drugs = []
        self.metadata = {}
        self.name_index = {}
        self.id_index = {}
        self.openfda_drugs = {}
        self.openfda_name_index = {}
        
        data = None
        
        # Try local file first
        if self.db_file:
            print(f"Loading drug database from {self.db_file}")
            try:
                with open(self.db_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load local database: {e}")
                data = None
        
        # If no local file, connect directly to Google Drive
        #MANP if data is None:
        #MANP     print("No local database found. Connecting to Google Drive...")
        #MANP     data = self._load_from_gdrive()
        
        if data:
            self.metadata = data.get('metadata', {})
            self.drugs = data.get('drugs', [])
            
            # Create indices
            for i, drug in enumerate(self.drugs):
                name = drug.get('name', '').lower()
                if name:
                    self.name_index[name] = i
                
                drug_id = drug.get('drugbank_ids', {}).get('primary')
                if drug_id:
                    self.id_index[drug_id] = i
            
            print(f"Loaded {len(self.drugs)} drugs")
        else:
            print("Warning: No database available. Using empty database.")
        
        # Load OpenFDA data if available
        openfda_enabled = os.getenv('ENABLE_OPENFDA_DATA', 'true').lower() not in ('0', 'false', 'no')
        if openfda_enabled and IJSON_AVAILABLE:
            self._load_openfda_data(openfda_file)
        
        DrugService._initialized = True

    def _load_openfda_data(self, openfda_file: str) -> None:
        """Stream OpenFDA dataset and keep only lightweight dosing/index info."""
        openfda_paths = [
            openfda_file,
            os.path.join('data', openfda_file),
            os.path.join('..', 'drug-app', openfda_file),
        ]
        
        actual_path = None
        for path in openfda_paths:
            if os.path.exists(path):
                actual_path = path
                break
        
        if not actual_path:
            return
        
        try:
            with open(actual_path, 'rb') as f:
                for drug_id, raw_entry in ijson.kvitems(f, 'drugs'):
                    simplified = self._simplify_openfda_entry(raw_entry)
                    if simplified:
                        self.openfda_drugs[drug_id] = simplified
                        self._index_openfda_name(simplified.get('drug_name'), drug_id)
                        for name in simplified.get('generic_names', []):
                            self._index_openfda_name(name, drug_id)
                        for name in simplified.get('brand_names', []):
                            self._index_openfda_name(name, drug_id)
        except Exception as exc:
            print(f"Warning: Failed to load OpenFDA data: {exc}")
            self.openfda_drugs = {}
            self.openfda_name_index = {}

    def _simplify_openfda_entry(self, raw_entry: Dict) -> Optional[Dict]:
        """Reduce OpenFDA entry to essential fields to stay within memory limits."""
        if not raw_entry:
            return None
        openfda_data = raw_entry.get('openfda_data', {})
        parsed_dosing = openfda_data.get('parsed_dosing', {}) or {}
        openfda_meta = openfda_data.get('openfda', {}) or {}
        
        def _as_list(values):
            if isinstance(values, list):
                return values
            if isinstance(values, str):
                return [values]
            return []
        
        def _clean_list(values):
            return [str(v).strip() for v in _as_list(values) if isinstance(v, str) and v.strip()]
        
        simplified = {
            'drug_name': raw_entry.get('drug_name', '').strip(),
            'generic_names': _clean_list(openfda_meta.get('generic_name', [])),
            'brand_names': _clean_list(openfda_meta.get('brand_name', [])),
            'parsed_dosing': {
                'frequency': parsed_dosing.get('frequency'),
                'times_per_day': parsed_dosing.get('times_per_day'),
                'times_per_day_range': parsed_dosing.get('times_per_day_range'),
                'routes': parsed_dosing.get('routes') or _clean_list(openfda_meta.get('route', [])),
                'route': parsed_dosing.get('route'),
                'instructions': parsed_dosing.get('instructions'),
                'has_dosing': parsed_dosing.get('has_dosing'),
                'source': parsed_dosing.get('source'),
            }
        }
        if not any(simplified['parsed_dosing'].values()):
            return None
        return simplified

    def _index_openfda_name(self, name: Optional[str], drug_id: str) -> None:
        if not name:
            return
        key = name.lower().strip()
        if not key:
            return
        self.openfda_name_index.setdefault(key, drug_id)

    def _search_openfda_partial(self, query_lower: str) -> Optional[Dict]:
        for entry in self.openfda_drugs.values():
            stored = entry.get('drug_name', '').lower()
            if query_lower in stored or stored.startswith(query_lower):
                return entry
        return None

    def _get_openfda_dosing(self, drug_name: str) -> Optional[Dict]:
        if not self.openfda_drugs:
            return None
        query_lower = drug_name.lower().strip()
        entry_id = self.openfda_name_index.get(query_lower)
        entry = self.openfda_drugs.get(entry_id) if entry_id else None
        if not entry:
            entry = self._search_openfda_partial(query_lower)
        if not entry:
            return None
        return entry.get('parsed_dosing', {})

    @staticmethod
    def _normalize_category_name(category_entry) -> str:
        """Normalize category entries that may be dicts or strings."""
        if isinstance(category_entry, dict):
            for key in ('category', 'name', 'mesh_id'):
                value = category_entry.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
        elif isinstance(category_entry, str):
            return category_entry.strip()
        elif category_entry is not None:
            text = str(category_entry).strip()
            if text:
                return text
        return ''

    def _dedupe_forms(self, forms: List[str]) -> List[str]:
        """Deduplicate forms, keeping unique base forms only."""
        if not forms:
            return []
        
        # First, normalize all forms
        normalized = set()
        for form in forms:
            if not form:
                continue
            # Normalize: title case, strip whitespace
            form = form.strip().title()
            # Skip if too long (likely multiple forms concatenated)
            if len(form) > 50:
                continue
            normalized.add(form)
        
        # Group by base form (first word before comma/semicolon)
        base_forms = {}
        for form in normalized:
            # Get base form
            base = re.split(r'[,;]', form)[0].strip().lower()
            # Keep shortest clean version for each base
            if base not in base_forms:
                base_forms[base] = form
            elif len(form) < len(base_forms[base]):
                base_forms[base] = form
        
        # Return unique forms, sorted by length (shortest first)
        result = sorted(base_forms.values(), key=len)
        return result[:4]

    def search_drugs(self, query: str) -> List[str]:
        """Search drugs by partial name match"""
        query_lower = query.lower()
        results = []
        for name in self.name_index.keys():
            if query_lower in name:
                idx = self.name_index[name]
                results.append(self.drugs[idx].get('name'))
        return sorted(results)[:50]

    def get_all_categories(self) -> List[str]:
        """Get all unique drug categories"""
        categories = set()
        for drug in self.drugs:
            for cat in drug.get('categories', []):
                cat_name = self._normalize_category_name(cat)
                if cat_name:
                    categories.add(cat_name)
        return sorted(list(categories))

    def get_drugs_by_category(self, category: str) -> List[str]:
        """Get all drugs in a specific category"""
        results = []
        category_lower = category.lower()
        for drug in self.drugs:
            for cat in drug.get('categories', []):
                cat_name = self._normalize_category_name(cat).lower()
                if not cat_name:
                    continue
                if category_lower in cat_name or cat_name in category_lower:
                    results.append(drug.get('name'))
                    break
        return sorted(list(set(results)))

    def find_drug(self, drug_name: str) -> Optional[Dict]:
        """Find drug by name"""
        name_lower = drug_name.lower()
        if name_lower in self.name_index:
            return self.drugs[self.name_index[name_lower]]
        return None

    def get_summary(self, drug_name: str) -> Dict:
        """Get drug summary with all data, including OpenFDA data"""
        drug = self.find_drug(drug_name)
        if not drug:
            return {'error': f"Drug '{drug_name}' not found"}
        
        dosing = drug.get('dosing_info', {})
        openfda_dosing_data = self._get_openfda_dosing(drug_name)
        
        frequency = dosing.get('frequency')
        times_per_day = dosing.get('times_per_day')
        routes = dosing.get('routes')
        
        if openfda_dosing_data:
            if not frequency:
                frequency = openfda_dosing_data.get('frequency')
            if not times_per_day:
                times_per_day = openfda_dosing_data.get('times_per_day_range') or openfda_dosing_data.get('times_per_day')
            if not routes:
                routes = openfda_dosing_data.get('routes', [])
                if not routes and openfda_dosing_data.get('route'):
                    routes = [openfda_dosing_data['route']] if isinstance(openfda_dosing_data['route'], str) else []
        
        openfda_full = dosing.get('openfda_full', {})
        if openfda_full:
            if not frequency:
                frequency = openfda_full.get('frequency')
            if not times_per_day:
                times_per_day = openfda_full.get('times_per_day_range') or openfda_full.get('times_per_day')
            if not routes:
                route = openfda_full.get('route')
                if route:
                    routes = [route] if isinstance(route, str) else route
        
        instructions = dosing.get('instructions', '')
        if not instructions and openfda_dosing_data:
            instructions = openfda_dosing_data.get('instructions', '')
        if not instructions and openfda_full:
            instructions = openfda_full.get('instructions', '')
        
        # Extract frequency from instructions with improved patterns
        if not frequency and instructions:
            instructions_lower = instructions.lower()
            # Once daily patterns (including hyphenated)
            if re.search(r'\b(?:once[\s-]+(?:a\s+)?dai?ly|q\.?d\.?\b|qd\b|\bonce[\s-]+a[\s-]+day)', instructions_lower):
                frequency = 'Once daily'
                times_per_day = '1'
            # Twice daily
            elif re.search(r'\b(?:twice[\s-]+(?:a\s+)?dai?ly|b\.?i\.?d\.?\b|bid\b|twice[\s-]+a[\s-]+day)', instructions_lower):
                frequency = 'Twice daily'
                times_per_day = '2'
            # Three times daily
            elif re.search(r'\b(?:three\s+times[\s-]+(?:a\s+)?dai?ly|t\.?i\.?d\.?\b|tid\b)', instructions_lower):
                frequency = 'Three times daily'
                times_per_day = '3'
            # Four times daily
            elif re.search(r'\b(?:four\s+times[\s-]+(?:a\s+)?dai?ly|q\.?i\.?d\.?\b|qid\b)', instructions_lower):
                frequency = 'Four times daily'
                times_per_day = '4'
            # Weekly
            elif re.search(r'\b(?:once[\s-]+(?:a\s+)?week(?:ly)?|weekly|every\s+week)', instructions_lower):
                frequency = 'Once weekly'
                times_per_day = '0.14'
            # Every X hours
            else:
                match = re.search(r'\bevery\s+(\d+)\s*(?:hours?|hrs?)\b', instructions_lower)
                if match:
                    hours = int(match.group(1))
                    frequency = f'Every {hours} hours'
                    times_per_day = str(round(24 / hours, 1))
                else:
                    match = re.search(r'\bq(\d+)h\b', instructions_lower)
                    if match:
                        hours = int(match.group(1))
                        frequency = f'Every {hours} hours'
                        times_per_day = str(round(24 / hours, 1))
        
        # Extract routes from 'dosages' key if not found elsewhere
        dosages_list = drug.get('dosages', [])
        if not routes and dosages_list:
            routes_set = set()
            for d in dosages_list:
                if d.get('route'):
                    routes_set.add(normalize_route(d['route']))
            routes = list(routes_set)[:5]
        
        # Normalize all routes
        if routes:
            routes = [normalize_route(r) for r in routes]
        
        # Extract strength from 'dosages' if available (deduplicated)
        strengths_set = set()
        if dosages_list:
            for d in dosages_list:
                if d.get('strength'):
                    # Normalize: "100.0 mg" -> "100 mg", remove trailing .0
                    strength = d['strength'].strip()
                    strength = re.sub(r'(\d+)\.0(\s*mg)', r'\1\2', strength)
                    strengths_set.add(strength)
        strengths = sorted(list(strengths_set))[:5]
        
        # Extract categories
        categories = []
        for cat in drug.get('categories', []):
            cat_name = self._normalize_category_name(cat)
            if cat_name:
                categories.append(cat_name)
        
        summary = {
            'name': drug.get('name'),
            'drugbank_id': drug.get('drugbank_ids', {}).get('primary'),
            'drugbank_ids': drug.get('drugbank_ids', {}),
            'type': drug.get('type'),
            'groups': drug.get('groups', []),
            'description': drug.get('description', ''),
            'mechanism_of_action': drug.get('mechanism_of_action', ''),
            'categories': categories,
            'dosing': {
                'has_dosing': dosing.get('has_dosing', False) or bool(dosages_list),
                'source': dosing.get('source') or ('DrugBank' if dosages_list else None),
                'frequency': frequency,
                'times_per_day': times_per_day,
                'routes': routes if routes else [],
                'strengths': strengths if strengths else [],
                'forms': self._dedupe_forms([d.get('form') for d in dosages_list if d.get('form')]),
            },
            'food_interactions': drug.get('food_interactions', [])[:3],
            'interaction_count': len(drug.get('drug_interactions', [])),
            'interactions_list': drug.get('drug_interactions', []),
            'properties': {},
            'pharmacokinetics': {
                'half_life': drug.get('half_life'),
                'absorption': drug.get('absorption'),
                'metabolism': drug.get('metabolism')
            },
            'dosages': drug.get('dosages', [])
        }
        
        for prop in drug.get('experimental_properties', []):
            kind = prop.get('kind')
            if kind in ['Melting Point', 'Water Solubility', 'Molecular Weight', 'logP', 'pKa']:
                summary['properties'][kind] = prop.get('value')
        
        return summary

    def _check_drug_class_interactions(self, drug1: Dict, drug2: Dict, drug1_name: str, drug2_name: str) -> List[Dict]:
        """Check for known drug class interactions that may not be in the database."""
        interactions = []
        
        categories1 = []
        for cat in drug1.get('categories', []):
            normalized = self._normalize_category_name(cat)
            if normalized:
                categories1.append(normalized.lower())
        
        categories2 = []
        for cat in drug2.get('categories', []):
            normalized = self._normalize_category_name(cat)
            if normalized:
                categories2.append(normalized.lower())
        
        moa1 = (drug1.get('mechanism_of_action', '') or '').lower()
        moa2 = (drug2.get('mechanism_of_action', '') or '').lower()
        desc1 = (drug1.get('description', '') or '').lower()
        desc2 = (drug2.get('description', '') or '').lower()
        
        # Check for benzodiazepine + beta-blocker interaction
        benzodiazepine_keywords = ['benzodiazepine', 'alprazolam', 'diazepam', 'lorazepam', 'clonazepam', 
                                   'temazepam', 'oxazepam', 'chlordiazepoxide', 'midazolam']
        beta_blocker_keywords = ['beta-blocker', 'beta blocker', 'beta-adrenergic', 'nebivolol', 'propranolol',
                                 'metoprolol', 'atenolol', 'bisoprolol', 'carvedilol', 'labetalol']
        
        is_benzodiazepine = any(kw in ' '.join(categories1) or kw in moa1 or kw in desc1 
                               for kw in benzodiazepine_keywords) or \
                          any(kw in drug1_name.lower() for kw in ['alprazolam', 'xanax'])
        is_beta_blocker = any(kw in ' '.join(categories2) or kw in moa2 or kw in desc2 
                             for kw in beta_blocker_keywords) or \
                        any(kw in drug2_name.lower() for kw in ['nebivolol', 'propranolol', 'metoprolol'])
        
        if not is_benzodiazepine:
            is_benzodiazepine = any(kw in ' '.join(categories2) or kw in moa2 or kw in desc2 
                                   for kw in benzodiazepine_keywords) or \
                              any(kw in drug2_name.lower() for kw in ['alprazolam', 'xanax'])
        if not is_beta_blocker:
            is_beta_blocker = any(kw in ' '.join(categories1) or kw in moa1 or kw in desc1 
                                 for kw in beta_blocker_keywords) or \
                            any(kw in drug1_name.lower() for kw in ['nebivolol', 'propranolol', 'metoprolol'])
        
        if is_benzodiazepine and is_beta_blocker:
            interactions.append({
                'drug': drug2_name if is_beta_blocker and drug2_name.lower() in [kw for kw in beta_blocker_keywords if len(kw) > 5] else drug1_name,
                'description': 'Benzodiazepines and beta-blockers may have additive effects on blood pressure lowering and CNS depression.',
                'severity': 'moderate',
                'source': 'known_class_interaction'
            })
        
        # Check for CNS depressants combination
        cns_depressant_keywords = ['cns depressant', 'sedative', 'hypnotic', 'anxiolytic', 'opioid', 
                                   'barbiturate', 'alcohol', 'antihistamine']
        is_cns_depressant1 = any(kw in ' '.join(categories1) or kw in moa1 or kw in desc1 
                                for kw in cns_depressant_keywords)
        is_cns_depressant2 = any(kw in ' '.join(categories2) or kw in moa2 or kw in desc2 
                                for kw in cns_depressant_keywords)
        
        if is_cns_depressant1 and is_cns_depressant2 and not (is_benzodiazepine and is_beta_blocker):
            interactions.append({
                'drug': drug2_name,
                'description': 'Both medications may cause CNS depression. Combined use may increase risk of drowsiness, dizziness, and impaired coordination.',
                'severity': 'moderate',
                'source': 'known_class_interaction'
            })
        
        return interactions

    def check_compatibility(self, drug1_name: str, drug2_name: str) -> Dict:
        """Check 3D printing compatibility between two drugs"""
        result = {
            'drug1': drug1_name,
            'drug2': drug2_name,
            'compatible': True,
            'issues': [],
            'warnings': [],
            'recommendations': [],
            'drug1_data': None,
            'drug2_data': None,
            'interactions': [],
            'routes': {'drug1': [], 'drug2': [], 'common': []},
            'dosing': {'drug1': {}, 'drug2': {}}
        }
        
        drug1 = self.find_drug(drug1_name)
        drug2 = self.find_drug(drug2_name)
        
        if not drug1:
            result['issues'].append(f"Drug '{drug1_name}' not found in database")
            result['compatible'] = False
            return result
        
        if not drug2:
            result['issues'].append(f"Drug '{drug2_name}' not found in database")
            result['compatible'] = False
            return result
        
        result['drug1_data'] = self.get_summary(drug1_name)
        result['drug2_data'] = self.get_summary(drug2_name)
        
        # Check biologics
        if drug1.get('type') == 'biotech':
            result['issues'].append(f"{drug1_name} is a biologic (protein/antibody) - cannot be 3D printed with standard methods")
            result['compatible'] = False
        
        if drug2.get('type') == 'biotech':
            result['issues'].append(f"{drug2_name} is a biologic (protein/antibody) - cannot be 3D printed with standard methods")
            result['compatible'] = False
        
        # Get dosing info
        dosing1 = drug1.get('dosing_info', {})
        dosing2 = drug2.get('dosing_info', {})
        
        freq1, times1 = self._extract_dosing(dosing1, drug1_name)
        freq2, times2 = self._extract_dosing(dosing2, drug2_name)
        
        result['dosing'] = {
            'drug1': {'frequency': freq1, 'times_per_day': times1},
            'drug2': {'frequency': freq2, 'times_per_day': times2}
        }
        
        if (freq1 or times1) and (freq2 or times2):
            if times1 and times2 and times1 == times2:
                result['recommendations'].append(f"📊 Same dosing frequency: both {freq1 or 'N/A'} ({times1}x/day)")
            elif times1 and times2:
                result['warnings'].append(f"⚠️ Different dosing frequencies: {freq1 or 'N/A'} ({times1}x/day) vs {freq2 or 'N/A'} ({times2}x/day)")
                result['recommendations'].append("📊 Timed-release formulation or separate administration may be needed")
        
        # Collect routes
        routes1, routes2 = self._collect_routes(drug1, dosing1), self._collect_routes(drug2, dosing2)
        common = set(r.lower() for r in routes1) & set(r.lower() for r in routes2)
        
        result['routes'] = {
            'drug1': routes1,
            'drug2': routes2,
            'common': [r for r in routes1 if r.lower() in common]
        }
        
        # Check interactions
        drug2_id = drug2.get('drugbank_ids', {}).get('primary')
        drug1_id = drug1.get('drugbank_ids', {}).get('primary')
        drug1_secondary_ids = drug1.get('drugbank_ids', {}).get('secondary', [])
        drug2_secondary_ids = drug2.get('drugbank_ids', {}).get('secondary', [])
        
        interactions_found = []
        
        for interaction in drug1.get('drug_interactions', []):
            interaction_id = interaction.get('drugbank_id')
            if interaction_id == drug2_id or interaction_id in drug2_secondary_ids:
                desc = interaction.get('description', '')
                drug_name = interaction.get('name', drug2_name)
                severity = classify_severity_simple(desc)
                emoji = get_severity_emoji(severity)
                interactions_found.append({
                    'drug': drug_name,
                    'description': desc,
                    'severity': severity
                })
                if severity == 'severe':
                    result['issues'].append(f"{emoji} SEVERE: {desc}")
                    result['compatible'] = False
                elif severity == 'minor':
                    result['recommendations'].append(f"{emoji} Minor: {desc}")
                else:
                    result['warnings'].append(f"{emoji} Moderate: {desc}")
        
        for interaction in drug2.get('drug_interactions', []):
            interaction_id = interaction.get('drugbank_id')
            if interaction_id == drug1_id or interaction_id in drug1_secondary_ids:
                desc = interaction.get('description', '')
                drug_name = interaction.get('name', drug1_name)
                if not any(i['description'] == desc for i in interactions_found):
                    severity = classify_severity_simple(desc)
                    emoji = get_severity_emoji(severity)
                    interactions_found.append({
                        'drug': drug_name,
                        'description': desc,
                        'severity': severity
                    })
                    if severity == 'severe':
                        msg = f"{emoji} SEVERE: {desc}"
                        if msg not in result['issues']:
                            result['issues'].append(msg)
                            result['compatible'] = False
                    elif severity == 'minor':
                        msg = f"{emoji} Minor: {desc}"
                        if msg not in result['recommendations']:
                            result['recommendations'].append(msg)
                    else:
                        msg = f"{emoji} Moderate: {desc}"
                        if msg not in result['warnings']:
                            result['warnings'].append(msg)
        
        # Check class interactions
        known_class_interactions = self._check_drug_class_interactions(drug1, drug2, drug1_name, drug2_name)
        if known_class_interactions:
            for interaction in known_class_interactions:
                if not any(i.get('description', '').lower() == interaction['description'].lower() for i in interactions_found):
                    interactions_found.append(interaction)
                    result['warnings'].append(f"⚠️ Potential class-based interaction: {interaction['description']}")
        
        result['interactions'] = interactions_found
        
        if result['compatible']:
            result['recommendations'].append(f"📊 Drug types: {drug1.get('type', 'unknown')} + {drug2.get('type', 'unknown')}")
            
            if result['routes']['common']:
                result['recommendations'].append(f"🛣️ Common routes of administration: {', '.join(result['routes']['common'][:5])}")
            elif result['routes']['drug1'] or result['routes']['drug2']:
                routes1_str = ', '.join(result['routes']['drug1'][:3]) if result['routes']['drug1'] else 'Unknown'
                routes2_str = ', '.join(result['routes']['drug2'][:3]) if result['routes']['drug2'] else 'Unknown'
                result['recommendations'].append(f"🛣️ Routes: {drug1_name} ({routes1_str}) vs {drug2_name} ({routes2_str})")
        
        return result

    def _extract_dosing(self, dosing: Dict, drug_name: str) -> tuple:
        """Extract frequency and times_per_day with fallbacks"""
        frequency = dosing.get('frequency')
        times_per_day = dosing.get('times_per_day')
        
        openfda_dosing_data = self._get_openfda_dosing(drug_name)
        
        if openfda_dosing_data:
            if not frequency:
                frequency = openfda_dosing_data.get('frequency')
            if not times_per_day:
                times_per_day = openfda_dosing_data.get('times_per_day_range') or openfda_dosing_data.get('times_per_day')
        
        openfda_full = dosing.get('openfda_full', {})
        if openfda_full:
            if not frequency:
                frequency = openfda_full.get('frequency')
            if not times_per_day:
                times_per_day = openfda_full.get('times_per_day_range') or openfda_full.get('times_per_day')
        
        instructions = dosing.get('instructions', '')
        if not instructions and openfda_dosing_data:
            instructions = openfda_dosing_data.get('instructions', '')
        if not instructions and openfda_full:
            instructions = openfda_full.get('instructions', '')
        
        if not frequency and instructions:
            instructions_lower = instructions.lower()
            # Enhanced patterns for frequency extraction
            if re.search(r'\b(?:once[\s-]+(?:a\s+)?day|once[\s-]+daily|q\.?d\.?\b|qd\b|daily\s+dose)', instructions_lower):
                frequency = 'Once daily'
                times_per_day = '1'
            elif re.search(r'\b(?:twice[\s-]+(?:a\s+)?day|twice[\s-]+daily|b\.?i\.?d\.?\b|bid\b|two\s+times\s+daily)', instructions_lower):
                frequency = 'Twice daily'
                times_per_day = '2'
            elif re.search(r'\b(?:three\s+times[\s-]+(?:a\s+)?day|t\.?i\.?d\.?\b|tid\b)', instructions_lower):
                frequency = 'Three times daily'
                times_per_day = '3'
            elif re.search(r'\b(?:four\s+times[\s-]+(?:a\s+)?day|q\.?i\.?d\.?\b|qid\b)', instructions_lower):
                frequency = 'Four times daily'
                times_per_day = '4'
            elif re.search(r'\bevery\s+(\d+)\s+hours?\b', instructions_lower):
                match = re.search(r'\bevery\s+(\d+)\s+hours?\b', instructions_lower)
                hours = int(match.group(1))
                if hours == 24:
                    frequency = 'Once daily'
                    times_per_day = '1'
                elif hours == 12:
                    frequency = 'Twice daily'
                    times_per_day = '2'
                elif hours == 8:
                    frequency = 'Three times daily'
                    times_per_day = '3'
                elif hours == 6:
                    frequency = 'Four times daily'
                    times_per_day = '4'
                else:
                    times = 24 // hours if hours > 0 else None
                    frequency = f'Every {hours} hours'
                    times_per_day = str(times) if times else None
            elif re.search(r'\b(?:once\s+weekly|weekly|every\s+week|q\.?w\.?\b|qw\b)', instructions_lower):
                frequency = 'Weekly'
                times_per_day = None
        
        return frequency, times_per_day

    def _collect_routes(self, drug: Dict, dosing: Dict) -> List[str]:
        """Collect all routes of administration for a drug, normalized"""
        routes = set()
        
        if dosing.get('routes'):
            route_data = dosing.get('routes')
            if isinstance(route_data, list):
                for r in route_data:
                    routes.add(normalize_route(r))
            elif isinstance(route_data, str):
                routes.add(normalize_route(route_data))
        
        openfda_full = dosing.get('openfda_full', {})
        if openfda_full:
            if openfda_full.get('routes'):
                route_data = openfda_full.get('routes')
                if isinstance(route_data, list):
                    for r in route_data:
                        routes.add(normalize_route(r))
                elif isinstance(route_data, str):
                    routes.add(normalize_route(route_data))
            if openfda_full.get('route'):
                route = openfda_full.get('route')
                if isinstance(route, str):
                    routes.add(normalize_route(route))
        
        for d in drug.get('dosages', []):
            if d and d.get('route'):
                routes.add(normalize_route(d.get('route')))
        
        return sorted(list(routes))

    def get_database_info(self) -> Dict:
        """Get database metadata"""
        return {
            'total_drugs': len(self.drugs),
            'drugs_with_dosing': self.metadata.get('drugs_with_dosing', 0),
            'source': self.db_file or 'No database loaded'
        }

    def get_alternatives_from_category(self, drug_name: str) -> List[str]:
        """Get alternative drugs from the same category"""
        drug = self.find_drug(drug_name)
        if not drug:
            return []
        
        categories = drug.get('categories', [])
        if not categories:
            return []
        
        first_category = ''
        for entry in categories:
            first_category = self._normalize_category_name(entry)
            if first_category:
                break
        
        if not first_category:
            return []
        
        alternatives = self.get_drugs_by_category(first_category)
        alternatives = [a for a in alternatives if a.lower() != drug_name.lower()]
        return alternatives[:10]

    def find_compatible_alternatives(self, target_drug: str, category: str, limit: int = 10) -> List[Dict]:
        """
        Find drugs from a specific category that have NO interactions with the target drug.
        
        Args:
            target_drug: The drug to check compatibility against
            category: The category to search for alternatives in
            limit: Maximum number of alternatives to return
            
        Returns:
            List of compatible drug dictionaries with name and basic info
        """
        target = self.find_drug(target_drug)
        if not target:
            return []
        
        # Get target drug's interaction IDs for fast lookup
        target_id = target.get('drugbank_ids', {}).get('primary')
        target_secondary = set(target.get('drugbank_ids', {}).get('secondary', []))
        
        target_interaction_ids = set()
        for interaction in target.get('drug_interactions', []):
            int_id = interaction.get('drugbank_id')
            if int_id:
                target_interaction_ids.add(int_id)
        
        # Get all drugs in the category
        category_drugs = self.get_drugs_by_category(category)
        
        compatible = []
        for drug_name in category_drugs:
            # Skip the target drug itself
            if drug_name.lower() == target_drug.lower():
                continue
            
            drug = self.find_drug(drug_name)
            if not drug:
                continue
            
            drug_id = drug.get('drugbank_ids', {}).get('primary')
            drug_secondary = set(drug.get('drugbank_ids', {}).get('secondary', []))
            
            # Check if this drug has any interaction with target
            has_interaction = False
            
            # Check if drug's ID is in target's interactions
            if drug_id and drug_id in target_interaction_ids:
                has_interaction = True
            
            # Check if any of drug's secondary IDs are in target's interactions
            if not has_interaction and drug_secondary:
                if drug_secondary & target_interaction_ids:
                    has_interaction = True
            
            # Also check reverse - if target is in this drug's interactions
            if not has_interaction:
                for interaction in drug.get('drug_interactions', []):
                    int_id = interaction.get('drugbank_id')
                    if int_id == target_id or int_id in target_secondary:
                        has_interaction = True
                        break
            
            if not has_interaction:
                # Get basic info for this compatible drug
                dosing = drug.get('dosing_info', {})
                compatible.append({
                    'name': drug.get('name'),
                    'type': drug.get('type'),
                    'has_dosing': dosing.get('has_dosing', False),
                    'frequency': dosing.get('frequency'),
                    'routes': dosing.get('routes', [])[:3] if dosing.get('routes') else []
                })
                
                if len(compatible) >= limit:
                    break
        
        return compatible


# Singleton instance
_drug_service: Optional[DrugService] = None


def get_drug_service() -> DrugService:
    """Get or create the drug service singleton"""
    global _drug_service
    if _drug_service is None:
        _drug_service = DrugService()
    return _drug_service
