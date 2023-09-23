package com.tjjhtjh.memorise.domain.memo.repository;

import com.tjjhtjh.memorise.domain.memo.service.dto.response.MemoDetailResponse;
import com.tjjhtjh.memorise.domain.memo.service.dto.response.MemoResponse;

import java.util.List;
import java.util.Optional;

public interface MemoRepositoryCustom {
    List<MemoResponse>  findWrittenByMeOrOpenMemoOrTaggedMemo(Long itemSeq, Long userSeq);
    Optional<MemoDetailResponse> memoDetailResponse(Long memoId);
}
