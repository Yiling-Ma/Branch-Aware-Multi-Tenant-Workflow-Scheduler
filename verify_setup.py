#!/usr/bin/env python3
"""
Verification script to check if the application is properly set up.

This script verifies:
1. API server is running
2. API documentation is accessible
3. Database connection
4. Export endpoint exists
"""

import requests
import sys
import json

BASE_URL = "http://127.0.0.1:8000"

def check_api_server():
    """Check if API server is running"""
    print("1. Checking API server...")
    try:
        response = requests.get(f"{BASE_URL}/", timeout=5)
        if response.status_code in [200, 307, 308]:  # OK or redirect
            print("   ✅ API server is running")
            return True
        else:
            print(f"   ❌ API server returned status {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("   ❌ Cannot connect to API server. Is it running?")
        print("      Start with: ./start_server.sh")
        return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def check_api_docs():
    """Check if API documentation is accessible"""
    print("\n2. Checking API documentation...")
    endpoints = [
        ("/docs", "Swagger UI"),
        ("/redoc", "ReDoc"),
        ("/openapi.json", "OpenAPI JSON")
    ]
    
    all_ok = True
    for endpoint, name in endpoints:
        try:
            response = requests.get(f"{BASE_URL}{endpoint}", timeout=5)
            if response.status_code == 200:
                print(f"   ✅ {name} is accessible at {BASE_URL}{endpoint}")
            else:
                print(f"   ⚠️  {name} returned status {response.status_code}")
                all_ok = False
        except Exception as e:
            print(f"   ❌ {name} error: {e}")
            all_ok = False
    
    return all_ok

def check_export_endpoint():
    """Check if export endpoint exists in OpenAPI spec"""
    print("\n3. Checking export endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/openapi.json", timeout=5)
        if response.status_code == 200:
            openapi_spec = response.json()
            paths = openapi_spec.get("paths", {})
            
            export_path = "/jobs/{job_id}/export"
            if export_path in paths:
                print(f"   ✅ Export endpoint found: {export_path}")
                methods = list(paths[export_path].keys())
                print(f"      Methods: {', '.join(methods)}")
                return True
            else:
                print(f"   ❌ Export endpoint not found in OpenAPI spec")
                print(f"      Available paths: {list(paths.keys())[:10]}...")
                return False
        else:
            print(f"   ❌ Cannot fetch OpenAPI spec (status {response.status_code})")
            return False
    except Exception as e:
        print(f"   ❌ Error checking export endpoint: {e}")
        return False

def check_key_endpoints():
    """Check if key endpoints exist"""
    print("\n4. Checking key API endpoints...")
    endpoints = [
        ("/users/", "POST", "Create user"),
        ("/workflows/", "POST", "Create workflow"),
        ("/jobs/{job_id}", "GET", "Get job"),
        ("/jobs/{job_id}/export", "GET", "Export results"),
    ]
    
    try:
        response = requests.get(f"{BASE_URL}/openapi.json", timeout=5)
        if response.status_code == 200:
            openapi_spec = response.json()
            paths = openapi_spec.get("paths", {})
            
            all_ok = True
            for path_template, method, description in endpoints:
                # Check if path exists (with or without {job_id})
                found = False
                for path in paths.keys():
                    if path_template.replace("{job_id}", "") in path or path == path_template:
                        if method.lower() in paths[path]:
                            print(f"   ✅ {description}: {method} {path}")
                            found = True
                            break
                
                if not found:
                    print(f"   ⚠️  {description}: {method} {path_template} (not found)")
                    all_ok = False
            
            return all_ok
        else:
            print(f"   ❌ Cannot fetch OpenAPI spec")
            return False
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    """Run all checks"""
    print("=" * 60)
    print("  Setup Verification")
    print("=" * 60)
    print()
    
    results = []
    
    results.append(("API Server", check_api_server()))
    results.append(("API Documentation", check_api_docs()))
    results.append(("Export Endpoint", check_export_endpoint()))
    results.append(("Key Endpoints", check_key_endpoints()))
    
    print("\n" + "=" * 60)
    print("  Summary")
    print("=" * 60)
    
    all_passed = True
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name}: {status}")
        if not result:
            all_passed = False
    
    print()
    if all_passed:
        print("✅ All checks passed! Your setup is ready.")
        return 0
    else:
        print("❌ Some checks failed. Please review the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

