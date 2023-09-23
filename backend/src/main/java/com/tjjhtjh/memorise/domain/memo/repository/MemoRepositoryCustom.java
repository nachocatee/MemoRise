package com.tjjhtjh.memorise.domain.memo.repository;

import com.tjjhtjh.memorise.domain.memo.service.dto.response.MemoResponse;

import java.util.List;

public interface MemoRepositoryCustom {
    List<MemoResponse>  findWrittenByMeOrOpenMemoOrTaggedMemo(Long itemSeq, Long userSeq);
}
