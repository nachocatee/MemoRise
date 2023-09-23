package com.tjjhtjh.memorise.domain.memo.service.dto.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

@Getter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class MemoDetailResponse {

    private String content;
    private String nickname;
    private LocalDateTime updatedAt;
    private String file;

}
