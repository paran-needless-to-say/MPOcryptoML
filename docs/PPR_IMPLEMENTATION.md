# PPR 구현 설명

## 논문의 요구사항

### Algorithm 1: Multi-source PPR의 시작

논문 Algorithm 1 Line 2-4:

```pseudocode
foreach v_i ∈ V do
    if in-degree (v_i) = 0 then
        Add v_i to the list of source nodes
```

**핵심**: in-degree가 **0인 노드**만을 source nodes로 식별해야 함

## 현재 구현

### 수정된 부분 ✅

1. **in-degree 계산 추가** (`ppr.py`):

   ```python
   def _compute_degrees(self):
       self.in_degrees = {}
       self.out_degrees = {}
       for node in self.nodes:
           self.in_degrees[node] = self.graph.in_degree(node)
           self.out_degrees[node] = self.graph.out_degree(node)
   ```

2. **Source nodes 식별** (`ppr.py`):

   ```python
   def get_source_nodes(self) -> Set[str]:
       """Line 2-4: in-degree가 0인 노드를 source nodes로 식별"""
       source_nodes = set()
       for node in self.nodes:
           if self.in_degrees[node] == 0:
               source_nodes.add(node)
       return source_nodes
   ```

3. **main.py에서 사용** (`main.py`):

   ```python
   # Line 2-4: in-degree가 0인 노드를 source nodes로 식별
   source_nodes_all = ppr.get_source_nodes()

   # in-degree=0인 노드가 없으면 모든 노드를 source로 사용 (더미 데이터)
   if len(source_nodes_all) == 0:
       sample_nodes = graph_obj.nodes[:min(30, len(graph_obj.nodes))]
   ```

## 더미 데이터의 문제

### 현황

- **더미 그래프**: 모든 노드가 서로 연결되어 있음
- **결과**: in-degree가 0인 노드가 없음
- **해결책**: 더미 데이터의 경우 fallback 로직 사용

### 실제 데이터에서는

- 실제 암호화폐 거래 그래프는 in-degree=0인 노드 존재
- 예: 최초 거래를 시작한 주소들
- 논문의 논리가 정확히 작동할 것

## 구현 정확도

### ✅ 논문 준수

- in-degree 계산 추가
- source nodes 식별 로직 구현
- 논문의 Line 2-4 정확히 반영

### ⚠️ 더미 데이터 한계

- 실제로 in-degree=0인 노드 없음
- fallback으로 모든 노드 사용

## 다음 단계

1. **실제 데이터 사용**: Etherscan API로 수집
2. **올바른 source nodes 식별**: in-degree=0인 노드 확인 가능
3. **논문 결과 재현**: 정확한 논리로 구현 완료
