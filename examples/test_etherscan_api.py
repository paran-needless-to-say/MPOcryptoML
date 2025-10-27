"""
Etherscan API 테스트
"""
import requests
import time

API_KEY = "TZ66JXC2M8WST154TM3111MBRRX7X7UAF9"

def test_api(address):
    """특정 주소로 API 테스트"""
    # Try V1 first (some endpoints still work)
    url = "https://api.etherscan.io/api"
    
    # If V1 doesn't work, API will tell us to use V2
    # V2 endpoint: https://api.etherscan.io/v2/api
    
    params = {
        'module': 'account',
        'action': 'txlist',
        'address': address,
        'startblock': 0,
        'endblock': 99999999,
        'page': 1,
        'offset': 10,
        'sort': 'desc',
        'apikey': API_KEY
    }
    
    print(f"\nTesting address: {address}")
    print(f"API URL: {url}")
    print(f"Params: {params}")
    
    try:
        response = requests.get(url, params=params, timeout=30)
        data = response.json()
        
        print(f"Status: {data.get('status')}")
        print(f"Message: {data.get('message')}")
        print(f"Result: {data.get('result')[:2] if isinstance(data.get('result'), list) else data.get('result')}")
        
        if data.get('status') == '1' and isinstance(data.get('result'), list):
            print(f"✓ Got {len(data['result'])} transactions")
            if len(data['result']) > 0:
                tx = data['result'][0]
                print(f"  First TX: from={tx.get('from')[:10]}..., to={tx.get('to')[:10]}..., timestamp={tx.get('timeStamp')}")
        else:
            print("❌ No transactions found")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    # Kaggle에서 실제 주소 가져오기
    import pandas as pd
    df = pd.read_csv('./notebooks/ethereum-frauddetection-dataset/transaction_dataset.csv')
    
    # 첫 번째 주소 테스트
    test_address = df.iloc[0]['Address']
    
    print("="*70)
    print("Etherscan API Test")
    print("="*70)
    
    test_api(test_address)
    
    # 두 번째도
    if len(df) > 1:
        test_address2 = df.iloc[1]['Address']
        test_api(test_address2)
    
    print("\n" + "="*70)
    print("Test completed")
    print("="*70)

