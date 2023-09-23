package com.tjjhtjh.memorise.domain.memo.repository;

import com.tjjhtjh.memorise.domain.memo.service.dto.response.MemoResponse;

import java.util.List;

public interface MemoRepositoryCustom {
    List<MemoResponse> itemMemoListofOpen(Long itemSeq, Long userSeq);
    List<MemoResponse> itemMemoListofClosed(Long itemSeq, Long userSeq);
    List<MemoResponse> itemMemoListofRestictMe(Long itemSeq, Long userSeq);
}
