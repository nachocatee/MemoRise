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

    private  String itemName;
    private Long countMemo;

    public MemoCountResponse countResponse(String itemName, Long countMemo) {
        return MemoCountResponse.builder()
                .itemName(itemName)
                .countMemo(countMemo)
                .build();
    }
}
