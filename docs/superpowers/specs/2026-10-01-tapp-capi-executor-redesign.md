# TAPP C API / executor 再設計案

Status: **提案・未承認・未実装**。対象はC ABIとその実行資源。Rustの計算APIやGEMM kernelの再設計ではない。

## 1. 結論

**収縮のC ABIにはTAPPそのものを使い、Rayonは `TAPP_executor` の内部実装にする。**
TAPPに似た第二の独自収縮APIや、独立した公開thread-pool handleは作らない。
既存 `tensorprimitives-tapp` と `tprims-core` / `tprims-exec` を統合して再利用する。
BLAS・linalgなどTAPP未規定の操作は `tprims_*` 拡張として残す。

## 2. 現行実装と標準の差

- `crates/tprims-contract-capi/src/lib.rs:97,206`: DLPack + `dot_general`、A/B/Cの3 operand、固定output-axis順序。`C = alpha*contract(A,B)+beta*C`。
- `tensorprimitives/crates/tensorprimitives-tapp/src/lib.rs:697,830`: label-based A/B/C/D、descriptorとdata pointerの分離は実装済み。しかしlibrary handleとexecutorを無視し、`Plan::run_raw` を呼ぶ。
- 同ファイル `ExecutorState:354`: thread数フィールドは実行に接続されていない。precisionも無条件に無視している。
- `crates/tprims-core/src/exec.rs:116,217`: 明示的owned Rayon pool、budget、同期closeとworker joinは実装済み。
- `crates/tprims-exec/src/exec.rs:116,200`: 小仕事はcaller上、並列kernelだけinstall、同pool内のSPMDは拒否、broadcastの直列化は実装済み。

TAPP論文§3.3はexecutorをCPU/GPU等の実行資源と位置づける。Rayonを使うか、thread数をどう指定するかは標準の要求ではない。
論文ではhandleからexecutorを得る、destroyでpointerを渡す等の説明があるが、現行upstreamヘッダは
`TAPP_create_executor(TAPP_executor*)` と `TAPP_destroy_executor(TAPP_executor)`。
**ABIはヘッダに合わせ、論文の説明を根拠にprototypeを改変しない。**

照合元: `TAPPorg/reference-implementation` commit `77c32d744ee6d339f504620cc80b8679601669bc`。
このcommitのヘッダを基準として記録・テストし、moving mainはABI基準にしない。

## 3. オブジェクトの責任

| Object | 責任 | 持たせないもの |
| --- | --- | --- |
| `TAPP_handle` | provider / planning state。計画作成時に0を拒否し、live handleはcaller契約 | 暗黙のpool、未使用cache |
| `TAPP_tensor_info` | dtype・extents・signed element strides。計画はmetadataをsnapshot | tensor dataの所有権 |
| `TAPP_tensor_product` | A/B/C/Dのlabels・element ops・実行計画 | 特定executorへのbinding、thread所有権 |
| `TAPP_executor` | 実行資源、thread budget、pool単位のSPMD gate | tensor data、計画作成 |
| `TAPP_status` | optional実行status | error codeとの混同、必須async機構 |

同じplanをserial executorでもRayon executorでも再利用できる。
初版は同期実行。`status` が非nullなら0を書き、status objectのallocationはしない。
返り値の `TAPP_error` と `TAPP_check_success` / `TAPP_explain_error` を使う。
非zero error codeはprovider定義であり、標準共通の番号とは主張しない。

## 4. Rayon poolの扱い

### 4.1 C host

- `TAPP_create_executor(&exec)` は**serial**。workerを作らない。
- `exec == 0` もdefault serialとして扱う。これは論文のdefault executor推奨に沿う**tprims側の選択**であり、現行ヘッダの保証ではない。
- 並列実行は明示的なprovider拡張で作成する。executorがprivate Rayon poolを所有し、複数planで使い回す。
- `nthreads == 0` はerror。`nthreads == 1` はserialでworker数0。環境変数やavailable CPU数から幅を決めない。
- poolのphysical widthは作成時に固定。budget変更ではthreadを増減しない。
- public pool handleは別に作らない。複数planでexecutorを共有するだけで現在の用途を満たす。

拡張は `<tprims/tapp_ext.h>` に分離する。以下の名前は提案で、いずれもTAPP標準外。

```c
TAPP_error tprims_tapp_executor_create_rayon(
    TAPP_executor *out, size_t nthreads, const tprims_rayon_opts *opts);
TAPP_error tprims_tapp_executor_set_budget(TAPP_executor exec, size_t budget);
TAPP_error tprims_tapp_executor_get_threads(
    TAPP_executor exec, size_t *pool_size, size_t *budget);
```

`opts` は既存のstack-size指定を再利用。budgetは1以上、pool幅までclamp。
queryのpool_sizeはserialで0、budgetは1。queryでactual active widthやCPU affinityは保証しない。
budget変更は実行開始時にsnapshotされ、既に進行中のcallには適用しない。
標準のattribute APIに独自keyを追加する方式は、標準keyが未定義なので初版では使わない。

### 4.2 Rust host / 他言語runtime

- Rust hostは現在の `Pool::borrow` + `Exec` を使い、hostのpoolを借りる。新しいpoolを横に作らない。
- C ABIに `rayon::ThreadPool*` を渡す仕組みは作らない。Rust型layoutと別shared-libraryのRayon runtimeを跨ぐ契約にはできない。
- host-owned RayonのRust利用では**同じThreadPoolに対するPool wrapperを一つだけ共有**する。SPMD gateがwrapper単位なので、複製すると同期を失う。
- Julia/Python等の外側が並列実行する場合は、まずdefault serial executorを使う。
- host callback executorは今回の必須範囲に含めない。実装する場合はbarrier-free batch + serial innerから始める。任意のtask-submit callbackはSPMD co-schedulingを保証せず、faerのRayon並列実行にもそのまま使えない。

### 4.3 実行規則

1. caller上でvalidationと制御を行う。API全体を無条件に `pool.install` しない。
2. width 1の仕事はcaller上。並列kernelだけpoolに入る。
3. batchのouter並列と単一itemのinner並列は同じbudgetを使い、outer並列時はinner serial。
4. SPMD/barrierの仕事は既存 `Exec::broadcast`。通常のRayon taskをbarrier participantとして投入しない。
5. budget・active width・dispatch widthは別。active widthが小さくてもRayon broadcastは全poolをdispatchする。budgetはOS worker総数の上限指定ではない。
6. 同pool内からSPMDを呼ぶ場合はserialまたは既存のbarrier-free経路へfallback。threadを追加しない。
7. 同一pool上のSPMD実行をpool単位で直列化。通常の計画はimmutableで、異なるoutputに対する並行実行を可能とする。budgetはcallごとの上限であり、複数call間の予約・公平性は保証しない。
8. TAPP経路は `Plan::run_raw_with` / `Spmd` seamを使用し、環境依存の `run_raw` に戻らない。

thread閾値やproviderの優劣は今回変更しない。Rayon採用による速度改善は未測定であり主張しない。

### 4.4 destruction

`TAPP_destroy_executor(exec)` に既存closeの同期joinを統合する。

- owned pool: callがなければpoolを停止し、全workerのjoinとTLS teardown後にhandleを解放。
- 実行中: provider定義のBUSY error、handleはliveのまま。完了後にretry可能。
- 同pool workerからdestroy: WOULD_DEADLOCK error、handleはliveのまま。
- default executor 0のdestroyはsuccess/no-op。
- live handleのdestroyは成功した一度だけ。二重destroy・stale/foreign handleは非対応。
- **呼び出し元は新規call開始とdestroyを同期する。** BUSY検出は任意のraw handleとdestroyのraceを安全にする保証ではない。

公開retain/release・独立close・detached reaperはこのTAPP executor契約では不要。
C/C++/Julia bindingがexecutorの所有者を持ち、borrowerより長く生存させる。
Rustのborrowed poolはRust lifetimeで管理し、tprimsはhost threadsを停止しない。

## 5. 収縮ABIとDLPack

TAPP標準のsignatureをそのまま提供する:

- descriptor作成はdata pointer不要。
- `TAPP_create_tensor_product`: 任意のlabel-based出力順序、A/B/C/Dのelement ops。
- `TAPP_execute_product`: scalarとraw data pointersだけを渡す。
- CとDは別bufferを許す。C == Dのin-place accumulationは、label順序を対応づけた後のaddress mappingが同一の場合のみ許す。同一pointerでも異なるmappingや部分重複は拒否。DとA/Bの重複も拒否。
- `beta == 0` はC/Dの旧値を読まない。C == NULLかつbeta != 0は曖昧な `TAPP_IN_PLACE` の意味を推測せずunsupported。
- pointer-arrayの `TAPP_execute_batched_product` とplan内のHadamard/batch labelは別物。
- batchは前処理をまとめ、各itemから再度exported C entryを呼ばずinternal executeを使う。validation errorは可能な範囲で書き込み前に検出する。実行途中の失敗には全batchのrollbackを保証しない。

初版は既存tensorcontractのlabel-based Planを再利用し、cases 1–4とf32/f64/c32/c64を維持。
precisionは `TAPP_DEFAULT_PREC` とstorageに一致する対応済み指定を受け入れる。
対応しないdtype・mixed storage・precision・element opは明示的error。precisionを黙って無視しない。
`op_D` を含む意味は既存実装とupstream oracleで確認する。全caseをdot_generalに無理に押し込まない。
既存Rust `tprims-contract::ContractPlan` とそのpermute+GEMM経路は維持する。
TAPPへの同経路追加は、label/C/D semanticsを満たす条件の整理と比較測定後の別作業。

DLPackは廃止しないが、TAPP標準signatureには混ぜない。
BLAS・将来linalgの `tprims_*` 拡張はDLPackを保持し、executorは共通の `TAPP_executor` に統一する。
TAPPを呼ぶbindingはdtype/device/read-onlyとoffsetを事前確認し、logical origin pointerを渡す。
raw TAPP ABIにはallocation lengthやDLPack read-only flagがなく、到達範囲の有効なmemoryはcallerの責任。
metadata由来のoverflow・output aliasing等は可能な範囲でABI側でも検証する。zero-copyを維持する。

## 6. 実装範囲と移行

一つのdeliverableにまとめ、作業順序だけ分ける。

1. **基準ABIを固定**: upstream commitのヘッダでC/C++ consumerをcompile/linkして確認。標準と拡張ヘッダを分離し、error・executor定義のownerを `tprims-core` に集約。
2. **executorを接続**: `tensorprimitives-tapp` の公開 `ExecutorState` を廃止し、`tprims-core` の共通executorと `tprims-exec` / `Spmd` seamで実行。owned poolのjoin処理を再利用。
3. **bundleに統合**: imported TAPP crateをrlibとして `tprims-bundle` のfeatureで選択。TAPP提供物も一つの `libtprims` にリンク。別々のTAPP/tprims shared library間でhandleを共有しない。
4. **C callerを移行**: old `tprims_contract_plan_*` をTAPPに移し、BLASのexecutor型とerror bridgeを統一。C例・ABI tests・C benchmarkを更新。
5. **旧surfaceを整理**: 現行tprims ABIは未安定なので、不要になった独自収縮APIとexecutor create/close/retain/release APIは互換shimを増やさず削除。公開済みconsumerの互換維持要求が判明した場合だけ期限付き移行を再検討。
6. README・architecture・design-principles・decision-logを承認された新方針に更新。現時点ではこの提案だけを保存し、既存のDecided行は変更しない。

新しいkernel、scheduler、cache、async runtime、公開pool objectは追加しない。

## 7. 受け入れ条件

- pinned upstreamの標準ヘッダのみをincludeするC/C++ programが、拡張なしでserial収縮できる。
- metadataのみでplan作成でき、同じplanを異なるdata pointers・serial/4T executorで再利用できる。
- cases 1–4、任意output順序、各operandのconjugation、C != D / C == D、negative strides、scalar、beta == 0を数値検証。C/Dの異なるlayoutも既存対応範囲で維持。
- unsupported dtype/precision/op、shape/stride overflow、aliasingはerror。通常callとbatchで検証を揃える。全C entryでpanicをABI外へ出さない。
- default/1Tでworker生成・pool entryなし。大きいpoolを指定しても小仕事はpool entryなし。`RAYON_NUM_THREADS` / `TENSORCONTRACT_THREADS` に関係なく実効幅がexecutorで決まる。
- 4T executorの複数plan共有でpoolは一つ。nested / concurrent SPMD、budget不足でdeadlock・追加threadなし。期限付きテストで検証。
- destructionのBUSY・self-worker errorはhandleを残し、成功時はworker TLS teardownまで完了。失敗したpool作成も開始済みworkerを回収する。
- bundleのTAPP/BLAS双方で同じexecutorを使える。別providerのhandle混用は非対応と明記。
- 同一timed boundaryで旧C ABI/新TAPP ABI/direct Rustのoverheadを**明示1T**で比較し、4T throughputを別に確認。effective width、provider/build/hardware、A/A noiseを記録。閾値は実装前に固定し、未測定のspeedupは主張しない。
- focused ABI/executor tests後にrepository-local gate。設計だけの今回、build/test/benchmarkは実行していない。

## References

- [TAPP paper, §3.3](https://arxiv.org/html/2601.07827v1#S3.SS3).
- [pinned executor.h](https://github.com/TAPPorg/reference-implementation/blob/77c32d744ee6d339f504620cc80b8679601669bc/api/include/tapp/executor.h).
- [pinned product.h](https://github.com/TAPPorg/reference-implementation/blob/77c32d744ee6d339f504620cc80b8679601669bc/api/include/tapp/product.h).
- `PERFORMANCE_TIPS.md`, CPU Threading Contract。
- `tensorprimitives/docs/refuted.md`, “rayon for the intra-contraction path”: barrier-bearing tasksとbroadcastを区別する理由。
