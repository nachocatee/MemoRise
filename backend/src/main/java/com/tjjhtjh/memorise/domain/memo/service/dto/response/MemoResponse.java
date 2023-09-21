package com.tjjhtjh.memorise.domain.memo.service.dto.response;

import com.tjjhtjh.memorise.domain.item.repository.entity.Item;
import com.tjjhtjh.memorise.domain.memo.repository.entity.AccessType;
import com.tjjhtjh.memorise.domain.memo.repository.entity.Memo;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;


@Getter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class MemoResponse {

    private String content;
    private Item item;
    private AccessType accessType;
    private String userId;
    private LocalDateTime updatedAt;
    private boolean bookmark;

    public MemoResponse response(Memo memo , Item item1) {
        return MemoResponse.builder()
                .content(memo.getContent())
                .item(item1)
                .accessType(memo.getAccessType())
                .updatedAt(memo.getUpdatedAt())
                .userId(memo.getUser().getEmail())
                .build();
    }

}
