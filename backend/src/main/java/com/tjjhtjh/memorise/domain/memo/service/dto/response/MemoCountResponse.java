package com.tjjhtjh.memorise.domain.memo.service.dto.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class MemoCountResponse {

    private Long itemSeq;
    private Long countMemo;

    public MemoCountResponse countResponse(Long itemSeq, Long countMemo) {
        return MemoCountResponse.builder()
                .itemSeq(itemSeq)
                .countMemo(countMemo)
                .build();
    }
}
