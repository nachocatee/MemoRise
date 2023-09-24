package com.tjjhtjh.memorise.domain.memo.service.dto.response;

import com.tjjhtjh.memorise.domain.memo.repository.entity.AccessType;
import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Getter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class MyMemoResponse {

    private String content;
    private AccessType accessType;
    private String nickname;
    private LocalDateTime updatedAt;
    private String file;
    private String itemImage;

}
