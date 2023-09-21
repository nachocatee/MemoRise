package com.tjjhtjh.memorise.domain.item.service.dto.request;

import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor
public class RegistItemRequest {
    private Long itemSeq;
    private String itemImage;
}
