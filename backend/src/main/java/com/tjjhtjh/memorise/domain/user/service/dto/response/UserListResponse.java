package com.tjjhtjh.memorise.domain.user.service.dto.response;

import lombok.Builder;
import lombok.Getter;
import lombok.Setter;

@Setter
@Getter
public class UserListResponse {

    Long userSeq;
    String nickname;
    String email;

    @Builder
    public UserListResponse(Long userSeq, String nickname, String email) {
        this.userSeq = userSeq;
        this.nickname = nickname;
        this.email = email;
    }

}
