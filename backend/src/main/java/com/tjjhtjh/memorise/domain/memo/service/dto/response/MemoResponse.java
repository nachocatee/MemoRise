package com.tjjhtjh.memorise.domain.memo.service.dto.response;

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
    private AccessType accessType;
    private String nickname;
    private LocalDateTime updatedAt;
    private String file;

    public MemoResponse itemMemoResponse(Memo memo) {
        return MemoResponse.builder()
                .content(memo.getContent())
                .accessType(memo.getAccessType())
                .updatedAt(memo.getUpdatedAt())
                .nickname(memo.getUser().getNickname())
                .file(memo.getFile())
                .build();
    }

}
