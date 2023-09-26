package com.tjjhtjh.memorise.domain.tag.service.dto.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class TaggedUserResponse {
    private String nickname;

    public TaggedUserResponse nicknameResponse(String nickname) {
        return TaggedUserResponse.builder()
                .nickname(nickname)
                .build();
    }
}
