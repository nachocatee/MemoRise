package com.tjjhtjh.memorise.domain.memo.repository;

import com.tjjhtjh.memorise.domain.memo.service.dto.request.MemoCountRequest;
import com.tjjhtjh.memorise.domain.memo.service.dto.response.MemoDetailResponse;
import com.tjjhtjh.memorise.domain.memo.service.dto.response.MemoResponse;
import com.tjjhtjh.memorise.domain.memo.service.dto.response.MyMemoResponse;

import java.util.List;
import java.util.Optional;

public interface MemoRepositoryCustom {
    List<MemoResponse>  findWrittenByMeOrOpenMemoOrTaggedMemo(Long itemSeq, Long userSeq);
    Optional<MemoDetailResponse> detailMemo(Long memoId);
    List<MyMemoResponse> findByMyMemoIsDeletedFalse(Long userSeq);
    List<MyMemoResponse> findByAllMyMemoIsDeletedFalse(Long userSeq);
    Long countMemoOfItem(Long itemSeq, Long userSeq);
}
