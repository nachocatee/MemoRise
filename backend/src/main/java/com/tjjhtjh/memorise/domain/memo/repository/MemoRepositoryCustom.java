package com.tjjhtjh.memorise.domain.memo.repository;

import com.tjjhtjh.memorise.domain.memo.repository.entity.Memo;
import com.tjjhtjh.memorise.domain.memo.service.dto.response.MemoResponse;

import java.util.List;

public interface MemoRepositoryCustom {
    List<Memo> itemMemoList(Long itemSeq, String email);
    MemoResponse findByMemoSeq(Long memoSeq, String email);
}
