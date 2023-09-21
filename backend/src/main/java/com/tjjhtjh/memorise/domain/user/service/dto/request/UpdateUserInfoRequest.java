package com.tjjhtjh.memorise.domain.user.service.dto.request;

import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor
public class UpdateUserInfoRequest {
    private String nickname;
    private String profile;
}