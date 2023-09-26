package com.tjjhtjh.memorise.domain.tag.service.dto.response;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Getter;
import lombok.NoArgsConstructor;

@Getter
@NoArgsConstructor
@AllArgsConstructor
@Builder
public class TaggedTeamResponse {

    private String name;

    public TaggedTeamResponse teamNameResponse(String teamName) {
        return TaggedTeamResponse.builder()
                .name(teamName)
                .build();
    }

}
